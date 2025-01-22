
import pandas as pd
from pathlib import Path
from datetime import datetime


# Configue for saving
data_date = datetime.now().strftime('%Y-%m-%d') 
date = datetime.now().strftime('%Y%m%d')
Path = Path(__file__).resolve().parent.parent



# Get paths
path_single_bike = Path / "output" / "single" / f"{data_date}_bikes.csv"
path_single_energy = Path / "output" / "single" / f"{data_date}_energy.csv"

# Read data
bike_forecast = pd.read_csv(path_single_bike)
energy_forecast = pd.read_csv(path_single_energy)

combined_forecast = pd.concat([energy_forecast, bike_forecast], ignore_index=True)

df = pd.DataFrame(
    {
        "forecast_date": data_date,
        "target": "no2",
        "horizon": ["36 hour", "40 hour", "44 hour", "60 hour", "64 hour", "68 hour"],
        "q0.025": "NA",
        "q0.25": "NA",
        "q0.5": "NA",
        "q0.75": "NA",
        "q0.975": "NA",
    }
)

combined_forecast = pd.concat([combined_forecast, df], ignore_index=True)

combined_forecast.to_csv(Path / "output" / f"{date}_RobbStark.csv", index=False, na_rep="NA")


