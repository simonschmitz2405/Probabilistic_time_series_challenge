import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date, timedelta
from tqdm import tqdm

class EnergyDataImporter:
    """Class to import energy data from the SMARD API.

    Attributes:
        url (str): URL to the SMARD API.
        energydata (pd.DataFrame): Dataframe to store the energy data.

    Functions:
        get_energy_data: Get the energy data on a hourly basis.
    """
    def __init__(self):
        self.url = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
        self.energydata = pd.DataFrame()

    def get_energy_data(self):
        """Get the energy data on a hourly basis.

        Returns:
            energydata: Get the energy with date as index and consumption as column.
        """
        # Get all available timestamps
        response = requests.get(self.url)
        timestamps = list(response.json()["timestamps"])[6 * 52:]  # Ignore the first 6 years

        col_names = ['date_time', 'Netzlast_Gesamt']
        
        # Loop over all available timestamps
        for stamp in tqdm(timestamps):
            dataurl = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{stamp}.json"
            response = requests.get(dataurl)
            rawdata = response.json()["series"]

            # Format timestamp and create DataFrame
            for i in range(len(rawdata)):
                rawdata[i][0] = datetime.fromtimestamp(int(str(rawdata[i][0])[:10])).strftime("%Y-%m-%d %H:%M:%S")

            self.energydata = pd.concat([self.energydata, pd.DataFrame(rawdata, columns=col_names)])

        self.energydata = self.energydata.dropna()
        self.energydata["date_time"] = pd.to_datetime(self.energydata.date_time, utc=True) + pd.DateOffset(hours=1)  
        self.energydata.set_index("date_time", inplace=True)
        self.energydata.rename(columns={"Netzlast_Gesamt": "consumption"}, inplace=True)
        self.energydata["consumption"] = self.energydata["consumption"] / 1000 


        return self.energydata