# Anomaly Detection in WTI Crude Oil Prices

West Texas Intermediate (WTI) is a primary benchmark for global crude oil. Large, unexpected moves in its daily price — the 2008 collapse, the 2015–2016 downturn, March 2022, or 20 April 2020, when the contract settled below zero for the first time — are the observations that matter for hedging and energy risk. This project asks whether those days can be recovered from the price history alone, without a labelled list of “events”.

The live, maintained pipeline is the standalone repository **[DrAdrianDC/WTI_Anomaly_Detection](https://github.com/DrAdrianDC/WTI_Anomaly_Detection)**. The `Production/` folder here is a snapshot of that repo.

The production model is an **LSTM autoencoder** on 10-day windows of locally vol-normalized ΔClose (USD/bbl). A day is flagged when reconstruction MSE exceeds the 99th percentile of *quiet* 2010–2019 error. That threshold is frozen. 2020–present is out of sample.

The figure uses the production model on the full WTI futures series (Yahoo Finance ticker `CL=F`). The 20 April 2020 negative print remains the standout; the same scoring also highlights late 2008, COVID-era March–April 2020, and the March 2022 spike.

![WTI close price with anomalous days highlighted](Production/output_results/plot-anomalies.png)

## How this repository is organised

| Folder | Contents |
| --- | --- |
| **[Production/](Production/)** | Snapshot of the standalone pipeline (training code, scores and figures). Prefer the live repo for Actions, Hub weights and weekly updates. |
| **[R&D/](R&D/)** | Original Jupyter notebook, Keras `.h5` weights and the first research plots. |

```bash
cd Production
python src/pipeline.py --mode evaluate
```

Setup, architecture and commands: [Production/README.md](Production/README.md).
Module-level reference: [Production/DOCUMENTATION.md](Production/DOCUMENTATION.md).
Live repo: [DrAdrianDC/WTI_Anomaly_Detection](https://github.com/DrAdrianDC/WTI_Anomaly_Detection).

## Data

Prices come from [yfinance](https://github.com/ranaroussi/yfinance) (`CL=F`, period `max`, unadjusted close).

## License

MIT. See [LICENSE](LICENSE).
