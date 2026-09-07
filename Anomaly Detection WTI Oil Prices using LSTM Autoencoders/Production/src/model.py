"""LSTM autoencoder for unsupervised anomaly detection on WTI prices.

Architecture (faithful to the research notebook, expressed in modern Keras)
--------------------------------------------------------------------------
The network is a symmetric sequence-to-sequence autoencoder. It is trained
to reconstruct a lookback window of scaled Close prices. Points whose
reconstruction error exceeds a statistical threshold are flagged as anomalies.

::

    Input                  (batch, T, 1)
      │
      ├─ LSTM(64, relu, return_sequences=True)
      ├─ Dropout(0.25)
      ├─ LSTM(32, relu, return_sequences=False)   ← latent bottleneck
      ├─ Dropout(0.25)
      ├─ RepeatVector(T)
      ├─ LSTM(32, relu, return_sequences=True)
      ├─ Dropout(0.25)
      ├─ LSTM(64, relu, return_sequences=True)
      ├─ Dropout(0.25)
      └─ TimeDistributed(Dense(1))
    Output                 (batch, T, 1)

Training objective
------------------
Mean squared error between the input window and its reconstruction (``mse``),
optimized with Adam. The notebook trained for up to 100 epochs with
``EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)``.

Anomaly score and threshold
---------------------------
Per-window reconstruction MSE is computed after inference. The decision
threshold is the 90th percentile of those scores (notebook default). MAE
is also reported as a training-quality metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model, Sequential

logger = logging.getLogger(__name__)


class ModelPersistenceError(RuntimeError):
    """Raised when a ``.keras`` artifact cannot be written or read."""


def set_global_seeds(seed: int) -> None:
    """Pin NumPy and TensorFlow RNGs for reproducible training runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


class LSTMAutoencoder:
    """Thin wrapper around the Keras LSTM autoencoder.

    Parameters
    ----------
    lookback:
        Window length ``T`` (notebook: 10 trading days).
    n_features:
        Number of input channels. Always ``1`` for univariate Close prices.
    encoder_units:
        Hidden sizes of the two encoder LSTM layers. Default ``(64, 32)``.
    decoder_units:
        Hidden sizes of the two decoder LSTM layers. Default ``(32, 64)``.
    dropout:
        Dropout rate applied after every LSTM. Default ``0.25``.
    activation:
        Recurrent activation used in the notebook: ``relu``.
    loss:
        Reconstruction loss. Default ``mse``.
    clipnorm:
        Optional global-norm clip on Adam to stabilize ReLU LSTMs.
    """

    def __init__(
        self,
        lookback: int = 10,
        n_features: int = 1,
        encoder_units: Sequence[int] = (64, 32),
        decoder_units: Sequence[int] = (32, 64),
        dropout: float = 0.25,
        activation: str = "relu",
        loss: str = "mse",
        optimizer: str = "adam",
        clipnorm: float = 1.0,
        threshold_percentile: float = 90.0,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2.")
        if n_features < 1:
            raise ValueError("n_features must be >= 1.")
        if len(encoder_units) != 2 or len(decoder_units) != 2:
            raise ValueError("This architecture expects two encoder and two decoder units.")

        self.lookback = lookback
        self.n_features = n_features
        self.encoder_units = tuple(int(u) for u in encoder_units)
        self.decoder_units = tuple(int(u) for u in decoder_units)
        self.dropout = dropout
        self.activation = activation
        self.loss = loss
        self.optimizer_name = optimizer
        self.clipnorm = clipnorm
        self.threshold_percentile = threshold_percentile
        self.model: Model | None = None
        self.threshold_: float | None = None

    def build(self) -> Sequential:
        """Construct and compile the Sequential LSTM autoencoder.

        The first layer is a Keras 3 ``Input`` so the model serializes
        cleanly to the native ``.keras`` format without the deprecated
        ``input_shape`` warning on the first LSTM.
        """
        enc_1, enc_2 = self.encoder_units
        dec_1, dec_2 = self.decoder_units

        model = Sequential(
            [
                Input(shape=(self.lookback, self.n_features), name="sequence_input"),
                LSTM(
                    enc_1,
                    activation=self.activation,
                    return_sequences=True,
                    name="encoder_lstm_1",
                ),
                Dropout(self.dropout, name="encoder_dropout_1"),
                LSTM(
                    enc_2,
                    activation=self.activation,
                    return_sequences=False,
                    name="encoder_lstm_2",
                ),
                Dropout(self.dropout, name="encoder_dropout_2"),
                RepeatVector(self.lookback, name="latent_repeat"),
                LSTM(
                    dec_1,
                    activation=self.activation,
                    return_sequences=True,
                    name="decoder_lstm_1",
                ),
                Dropout(self.dropout, name="decoder_dropout_1"),
                LSTM(
                    dec_2,
                    activation=self.activation,
                    return_sequences=True,
                    name="decoder_lstm_2",
                ),
                Dropout(self.dropout, name="decoder_dropout_2"),
                TimeDistributed(Dense(self.n_features), name="reconstruction"),
            ],
            name="wti_lstm_autoencoder",
        )
        model.compile(optimizer=self._make_optimizer(), loss=self.loss)
        self.model = model
        logger.info(
            "Built LSTM autoencoder: lookback=%s, encoder=%s, decoder=%s, dropout=%s.",
            self.lookback,
            self.encoder_units,
            self.decoder_units,
            self.dropout,
        )
        return model

    def fit(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        patience: int,
        shuffle: bool = False,
    ) -> History:
        """Train the autoencoder to reconstruct its own inputs.

        ``x_train`` / ``x_val`` must already be 3D:
        ``(n_samples, lookback, n_features)``. Time series are not shuffled
        by default so contiguous market regimes stay intact.
        """
        model = self._require_model()
        x_train = self._validate_tensor(x_train, "x_train")
        x_val = self._validate_tensor(x_val, "x_val")
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def reconstruct(self, sequences: np.ndarray) -> np.ndarray:
        """Return reconstructed windows with the same shape as ``sequences``."""
        model = self._require_model()
        tensor = self._validate_tensor(sequences, "sequences")
        predictions = model.predict(tensor, verbose=0)
        return np.asarray(predictions, dtype=np.float32)

    def reconstruction_errors(
        self,
        sequences: np.ndarray,
        metric: str = "mse",
    ) -> np.ndarray:
        """Per-window reconstruction error.

        Parameters
        ----------
        sequences:
            3D input tensor.
        metric:
            ``mse`` (threshold / anomaly score) or ``mae`` (quality gate).
        """
        reconstructed = self.reconstruct(sequences)
        flat_true = sequences.reshape(sequences.shape[0], -1)
        flat_pred = reconstructed.reshape(reconstructed.shape[0], -1)
        delta = flat_pred - flat_true
        if metric == "mse":
            return np.mean(np.square(delta), axis=1)
        if metric == "mae":
            return np.mean(np.abs(delta), axis=1)
        raise ValueError("metric must be 'mse' or 'mae'.")

    def mean_absolute_error(self, sequences: np.ndarray) -> float:
        """Mean MAE across windows — used as the retrain quality gate."""
        return float(np.mean(self.reconstruction_errors(sequences, metric="mae")))

    def fit_threshold(self, sequences: np.ndarray) -> float:
        """Set ``threshold_`` to the configured percentile of reconstruction MSE."""
        errors = self.reconstruction_errors(sequences, metric="mse")
        self.threshold_ = float(np.percentile(errors, self.threshold_percentile))
        logger.info(
            "Anomaly threshold set to P%s(MSE) = %.6f.",
            self.threshold_percentile,
            self.threshold_,
        )
        return self.threshold_

    def detect_anomalies(
        self,
        sequences: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a 0/1 flag per window (1 = reconstruction MSE above threshold)."""
        cutoff = self.threshold_ if threshold is None else threshold
        if cutoff is None:
            raise ValueError("No threshold provided. Call fit_threshold() first.")
        errors = self.reconstruction_errors(sequences, metric="mse")
        return np.where(errors > cutoff, 1, 0).astype(np.int32)

    def save(self, path: str | Path) -> Path:
        """Persist weights and graph in the native Keras v3 ``.keras`` format."""
        model = self._require_model()
        destination = Path(path)
        if destination.suffix != ".keras":
            raise ModelPersistenceError(
                f"Refusing to save to '{destination}'. Use a .keras path."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
        logger.info("Saved Keras model to %s", destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        threshold_percentile: float = 90.0,
        threshold: float | None = None,
    ) -> LSTMAutoencoder:
        """Load a previously saved ``.keras`` autoencoder."""
        source = Path(path)
        if not source.exists():
            raise ModelPersistenceError(f"Model artifact not found: {source}")
        try:
            loaded = keras.models.load_model(source)
        except Exception as exc:  # Keras raises a variety of I/O errors
            raise ModelPersistenceError(f"Failed to load {source}: {exc}") from exc

        lookback, n_features = cls._infer_input_shape(loaded)
        wrapper = cls(
            lookback=lookback,
            n_features=n_features,
            threshold_percentile=threshold_percentile,
        )
        wrapper.model = loaded
        wrapper.threshold_ = threshold
        logger.info("Loaded Keras model from %s (lookback=%s).", source, lookback)
        return wrapper

    def summary(self) -> None:
        """Print the Keras layer table (useful in local training logs)."""
        self._require_model().summary()

    def _make_optimizer(self) -> keras.optimizers.Optimizer:
        if self.optimizer_name.lower() != "adam":
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer_name}'. Only Adam is configured."
            )
        return keras.optimizers.Adam(clipnorm=self.clipnorm)

    def _require_model(self) -> Sequential | Model:
        if self.model is None:
            raise RuntimeError("Model is not built. Call build() or load() first.")
        return self.model

    def _validate_tensor(self, tensor: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(tensor, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(
                f"{name} must be 3D (n_samples, lookback, n_features); "
                f"received shape {array.shape}."
            )
        if array.shape[1] != self.lookback or array.shape[2] != self.n_features:
            raise ValueError(
                f"{name} shape {array.shape} does not match "
                f"(*, {self.lookback}, {self.n_features})."
            )
        return array

    @staticmethod
    def _infer_input_shape(model: Model) -> tuple[int, int]:
        shape = model.input_shape
        if not shape or len(shape) != 3:
            raise ModelPersistenceError(
                f"Unexpected input_shape {shape}; expected (None, lookback, features)."
            )
        lookback = int(shape[1])
        n_features = int(shape[2])
        return lookback, n_features


def keras_model_to_dict(model: Model) -> dict[str, Any]:
    """Small diagnostic payload for metadata.json."""
    return {
        "name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "trainable_params": int(model.count_params()),
    }
