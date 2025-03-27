# Probabilistic Time Series Forecasting

This repository contains the implementation developed for the **Probabilistic Time Series Challenge** held during the winter semester 2024/2025 at KIT. The challenge focused on real-time quantile forecasting for two key applications:

- **Bike counts in Karlsruhe** (7-day horizon forecasts)
- **Energy demand across Germany** (hour-specific forecasts over a 3-day horizon)

## Overview
Accurate forecasting of bike counts and energy demand has substantial practical value. Predicting bike usage in real-time can support urban planning, optimize bike-sharing systems, and promote sustainable transportation. Similarly, precise energy demand forecasting is essential for balancing supply and demand, maintaining grid stability, and facilitating the integration of renewable energy sources. 

However, these forecasting tasks are inherently complex due to factors such as:

- Weather variability
- Seasonal patterns
- External events impacting demand and usage

## Objectives
The primary goal of this project was to explore and implement various probabilistic forecasting techniques. Through iterative model refinement and collaborative learning, we aimed to develop accurate quantile predictions that account for uncertainty, rather than relying solely on point estimates.

## Acknowledgments
A special thank you to **Professor Dr. Fabian Krüger** and his team for organizing this challenge and providing valuable support throughout the semester.

## Repository Structure
```plaintext
├── evaluation/               # Contains necessary files for evaluation submission format checking over real-time forecasting.
├── models/                   # Implementation of various probabilistic forecasting models.
├── processing/               # Contains scripts for retrieving and preprocessing weather data, as well as implementing the validation strategy.
├── Main.ipynb                # Entry-Point for Forecasting.
├── hyperparameter.py         # Stores all hyperparameter for each model on quantile level.
├── hyperparameter_optuna.py  # Performes the hyperparameter search using optuna package.
├── visualization.ipynb       # Contains some usefull visualization.
├── README.md                 # This document.
```

## Usage
### Clone the repository
```sh
git clone git@github.com:simonschmitz2405/Probabilistic_time_series_challenge.git
```

### Run the forecasting models
Execute the scripts from the `Main.ipynb` file or explore the  `models/` folder for detailed analysis.





