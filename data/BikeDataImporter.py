import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import date

class BikeDataImporter:
    def __init__(self, start_date = '01/01/2019', station_id=100126474, organisme_id=4586):
        self.start_date = start_date
        self.station_id = station_id
        self.organisme_id = organisme_id
        self.df = None

    def get_bike_data(self):
        dataurl = (f"https://www.eco-visio.net/api/aladdin/1.0.0/pbl/publicwebpageplus/data/"
                   f"{self.station_id}?idOrganisme={self.organisme_id}&idPdc={self.station_id}"
                   f"&interval=4&flowIds={self.station_id}&debut={self.start_date}")
        response = requests.get(dataurl)
        rawdata = response.json()
        
        self.df = pd.DataFrame(rawdata, columns=['date', 'bike_count'])
        self.df['bike_count'] = self.df['bike_count'].astype(float)

        # Convert date column to datatime with UTC timezone
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        self.df.set_index('date', inplace=True)
        self.df = self.df.dropna(axis=0)
        return self.df

    def get_bike_data_hourly(self):
        dataurl = (f"https://www.eco-visio.net/api/aladdin/1.0.0/pbl/publicwebpageplus/data/"
                   f"{self.station_id}?idOrganisme={self.organisme_id}&idPdc={self.station_id}"
                   f"&interval=3&flowIds={self.station_id}&debut={self.start_date}")
        response = requests.get(dataurl)
        rawdata = response.json()
        
        self.df = pd.DataFrame(rawdata, columns=['date', 'bike_count'])
        self.df['bike_count'] = self.df['bike_count'].astype(float)
        self.df.set_index(pd.to_datetime(self.df['date'], utc=True), inplace=True)

        self.df.drop(columns=['date'], inplace=True)
        self.df = self.df.dropna(axis=0)
        return self.df
    

if __name__ == "__main__":
    bike_importer = BikeDataImporter()
    bike_data = bike_importer.get_bike_data()
    print(bike_data)