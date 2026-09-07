# Anomaly Detection on WTI Oil Prices (LSTM Autoencoder)

Unsupervised detection of abnormal daily WTI futures moves (`CL=F`).

This folder holds two stages of the same case:

| Folder | What it is |
| --- | --- |
| **[Production/](Production/)** | Modular pipeline, champion model, scores and plots. Start here. |
| **[R&D/](R&D/)** | Original research notebook, `.h5` weights and early figures. |

![Anomaly Detection WTI Oil Price](Production/output_results/plot-anomalies.png)

The production champion flags 20 April 2020 (WTI at −$37.63) as the strongest reconstruction error, and also clusters 2008, 2015–2016 and March 2022.

To run the pipeline:

```bash
cd Production
python src/pipeline.py --mode evaluate
```

Setup, architecture and MLOps notes are in [Production/README.md](Production/README.md). Module-level detail is in [Production/DOCUMENTATION.md](Production/DOCUMENTATION.md).
