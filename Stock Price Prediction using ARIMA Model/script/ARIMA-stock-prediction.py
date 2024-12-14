# ARIMA Stock Price Forecasting

# Import libraries
import yfinance as yf
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Download data
spy500 = yf.download("SPY", start="2011-01-01", end="2024-12-13")

# Save the downloaded data to a CSV file
spy500.to_csv('spy500_data.csv', index=True)  # Save with the index as the Date column

# Error Handling for empty data
if spy500.empty:
    print("Failed to download data. Please check the ticker or internet connection.")
    exit()

# Handle MultiIndex: Extract 'Close' for 'SPY'
if ('Close', 'SPY') not in spy500.columns:
    print("Error: ('Close', 'SPY') column not found in the data.")
    exit()

# Extract and preprocess 'Close' prices
dataset_spy500 = spy500[('Close', 'SPY')].copy()
dataset_spy500.name = 'Close'  # Rename for clarity
dataset_spy500 = dataset_spy500.dropna()

# Add dates as the index
dataset_spy500 = dataset_spy500.reset_index()
dataset_spy500['Date'] = pd.to_datetime(dataset_spy500['Date'])  # Ensure proper datetime format
dataset_spy500.set_index('Date', inplace=True)


# Auto ARIMA to select optimal ARIMA parameters
model_auto = auto_arima(dataset_spy500['Close'], seasonal=False, trace=True)
print(model_auto.summary())
optimal_order = model_auto.order

# Define the ARIMA model
def arima_forecast(history, order):
    model = ARIMA(history, order=order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    return yhat

# Split data into train and test sets
X = dataset_spy500.values
size = int(len(X) * 0.8)
train, test = X[:size], X[size:]


# Walk-forward validation
history = [x for x in train]
predictions = []
for t in range(len(test)):
    yhat = arima_forecast(history, optimal_order)
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

# Evaluate model
predictions = np.array(predictions).flatten()  # Flatten predictions
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f"Root Mean Squared Error: {rmse}")

# Plotting results
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(dataset_spy500.iloc[size:].index, test, label='Real')
plt.plot(dataset_spy500.iloc[size:].index, predictions, color='red', label='Predicted')
plt.title(f'S&P500 Stock: ARIMA Predictions vs Actual Values (RMSE: {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('arima_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

