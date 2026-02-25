# Probabilistic Time Series Forecasting Framework

This repository contains a probabilistic machine learning framework developed for the real-time forecasting of highly seasonal, non-linear time series. 

Unlike standard point-estimate forecasting, this engine focuses on strict uncertainty quantification via quantile regression, providing calibrated confidence intervals for decision-making under uncertainty. The framework was developed and evaluated across two distinct, high-noise domains:
- **High-Frequency Energy Demand:** Hour-specific forecasts across Germany over a 3-day horizon.
- **Urban Mobility (Bike Counts):** Daily aggregated forecasts over a 7-day horizon in Karlsruhe, factoring in exogenous weather variables.

## Architectural Overview

The forecasting engine utilizes a probabilistic ensemble approach to capture both short-term volatility and long-term seasonal dependencies. The core architecture includes:
- **Models:** Gradient Boosting Regressors, LightGBM, Multiple Seasonal-Trend Decomposition using LOESS (MSTL), and Linear Quantile Regression.
- **Optimization:** Bayesian hyperparameter optimization utilizing `Optuna` to systematically minimize Pinball Loss across all target quantiles.
- **Validation:** A rigorous 10-week rolling-window backtest designed to perfectly mimic real-time production deployment and prevent data leakage.

## Out-of-Sample Performance

Based on the 10-week rolling-window evaluation, the optimized ensemble strategy significantly outperformed baseline benchmarks:
- **Global D2 Score:** 0.48
- **Calibration:** Achieved the lowest empirical calibration errors for both high-frequency (energy) and low-frequency (mobility) demand targets.

## Repository Structure

```text
├── evaluation/               # Scripts for strict submission formatting and real-time evaluation checking
├── models/                   # Core implementations of the probabilistic forecasting algorithms
├── processing/               # Data ingestion, exogenous variable preprocessing (weather), and rolling-window logic
├── Main.ipynb                # Primary entry-point notebook for end-to-end forecasting execution
├── hyperparameter.py         # Static hyperparameter configurations across distinct quantile levels
├── hyperparameter_optuna.py  # Automated Bayesian search space execution (Optuna)
├── visualization.ipynb       # Diagnostic plots for probabilistic calibration and interval width analysis
└── README.md                 # Project documentation
```

## Setup and Execution

To execute the forecasting framework locally
1. Clone the repository
```bash
git clone [https://github.com/simonschmitz2405/Probabilistic_time_series_challenge.git](https://github.com/simonschmitz2405/Probabilistic_time_series_challenge.git)
cd Probabilistic_time_series_challenge
```
2. Set up the virtual enviroment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Execute the hyperparameter search
```bash
python hyperparameter_optuna.py
```

4. Run the forecasting pipeline

Open `Main.ipynb` in your preferred Juptyer enviroment to execute the data processing, model training, and quantile prediction generation.

## Acknowledgments
A special thank you to **Professor Dr. Fabian Krüger** and his team for organizing this challenge and providing valuable support throughout the semester.





