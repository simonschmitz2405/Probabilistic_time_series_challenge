from data.BikeDataImporter import BikeDataImporter
import pandas as pd
import holidays
from pathlib import Path


class DataPreparing:
    def get_bike_and_weather_data(self):
        """Get the bike data and the weather data as a single dataframe.

        Returns:
            data_w_weather (pd.DataFrame): The bike data with the weather data.
        """
        bike_data = self.get_bike_data()
        # Give the last date of the bike data
        weather_data_hourly = self.get_weather_data_hourly()
        weather_data_daily = self.get_weather_data_daily(weather_data_hourly)
        data_w_weather = self.combine_data_weather(bike_data, weather_data_daily)

        return data_w_weather

    def get_bike_data(self):
        """Call the BikeDataImporter to get the bike data.

        Args:
            start_date (str): The start date for the data.

        Returns:
            data (pd.Dataframe): Bike counts for each day.
        """
        bike_data = BikeDataImporter()
        data = bike_data.get_bike_data()
        return data

    def get_weather_data_hourly(self):
        """Get the weather data for Karlsruhe from the historical and forecast data on hourly basis.

        Returns:
            weather_data (pd.Dataframe): Hourly historical and forecast weather data.
        """
        todays_date = pd.Timestamp.now().strftime("%Y%m%d")
        # Load the historical and the forecast weather data
        path = Path(__file__).parent.parent  # Define the path to store the cach file
        history_data_path = (
            path
            / "data"
            / "weather"
            / "history"
            / "karlsruhe"
            / f"history_karlsruhe_{todays_date}.csv"
        )
        forecast_data_path = (
            path
            / "data"
            / "weather"
            / "forecasts"
            / "karlsruhe"
            / f"forecast_karlsruhe_{todays_date}.csv"
        )
        # history_data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "weather", "history", "karlsruhe", f"history_karlsruhe_{todays_date}.csv")
        # forecast_data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "weather", "forecasts", "karlsruhe", f"forecast_karlsruhe_{todays_date}.csv")

        # Read in the historical and forecast data
        history_data = pd.read_csv(history_data_path)
        forecast_data = pd.read_csv(forecast_data_path)

        # Set the index to the date
        history_data.set_index("date", inplace=True)
        forecast_data.set_index("date", inplace=True)

        # Convert the index to datetime
        history_data.index = pd.to_datetime(history_data.index)
        forecast_data.index = pd.to_datetime(forecast_data.index)

        # Convert the index to datetime
        history_data.index = history_data.index.tz_convert("UTC")
        forecast_data.index = forecast_data.index.tz_convert("UTC")

        # Convert the index to European timezone (e.g., Europe/Paris)
        # history_data.index = history_data.index.tz_convert('Europe/Paris')
        # forecast_data.index = forecast_data.index.tz_convert('Europe/Paris')

        # The history data is not complete, so we need to filter it
        max_hisory_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
        history_data_filtered = history_data.loc[
            (history_data.index < max_hisory_date)
        ].copy()
        forecast_data_filtered = forecast_data.loc[
            (forecast_data.index >= max_hisory_date)
        ].copy()

        # Merge the historical and forecast data
        weather_data = pd.concat(
            [history_data_filtered, forecast_data_filtered], axis=0
        )

        return weather_data.iloc[1:]

    def get_weather_data_daily(self, weather_data_hourly):
        """Resample the hourly weather data to daily data.

        Args:
            weather_data_hourly (pd.Dataframe): Hourly historical and forecast weather data.

        Returns:
            weather_data_daily (pd.Dataframe): Daily historical and forecast weather data.
        """
        weather_data_daily = weather_data_hourly.resample(
            "D"
        ).mean()  # Todo: Check if we need to resample differently
        return weather_data_daily

    def combine_data_weather(self, bike_data, weather_data):
        """Merge the bike data with the weather data.

        Args:
            bike_data (pd.Dataframe): Daily bike data.
            weather_data (pd.Dataframe): Daily weather data.

        Returns:
            bike_and_weather_data (pd.Dataframe): bike_and_weather_data
        """
        bike_and_weather_data = pd.merge(
            bike_data, weather_data, left_index=True, right_index=True, how="left"
        )
        return bike_and_weather_data

    def create_features(self, data):
        """Create the features based on the data.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = data.copy()

        # Create time features
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["quarter"] = data.index.quarter
        data["weekday"] = data.index.weekday

        # Create lag features
        target_map = data["bike_count"].to_dict()
        # data["lag_1"] = (data.index - pd.Timedelta(days=1)).map(target_map)
        # data["lag_2"] = (data.index - pd.Timedelta(days=2)).map(target_map)
        # data["lag_3"] = (data.index - pd.Timedelta(days=3)).map(target_map)
        # data["lag_4"] = (data.index - pd.Timedelta(days=4)).map(target_map)
        data["lag_5"] = (data.index - pd.Timedelta(days=7)).map(target_map)
        data["lag_6"] = (data.index - pd.Timedelta(days=14)).map(target_map)
        data["lag_7"] = (data.index - pd.Timedelta(days=20)).map(target_map)
        data["lag_8"] = (data.index - pd.Timedelta(days=28)).map(target_map)
        # data["lag_9"] = (data.index - pd.Timedelta(days=50)).map(target_map)

        # Create holiday feature
        data["is_holiday"] = 0
        bw_holidays = holidays.country_holidays("DE", subdiv="BW")
        for i, date in enumerate(data.index):
            if date in bw_holidays:
                data.at[date, "is_holiday"] = 1

        # Create corona feature
        lockdown1_start = pd.Timestamp("2020-03-16", tz="UTC")
        lockdown1_end = pd.Timestamp("2020-05-11", tz="UTC")
        easing_start = pd.Timestamp("2020-06-01", tz="UTC")
        easing_end = pd.Timestamp("2020-10-30", tz="UTC")
        lockdown2_start = pd.Timestamp("2020-11-02", tz="UTC")
        lockdown2_end = pd.Timestamp("2020-02-14", tz="UTC")
        vaccionation_start = pd.Timestamp("2021-03-01", tz="UTC")
        vaccionation_end = pd.Timestamp("2021-06-30", tz="UTC")

        data.loc[(data.index < lockdown1_start), "corona_phase"] = 4  # Pre-Pandemic
        data.loc[
            (data.index >= lockdown1_start) & (data.index <= lockdown1_end),
            "corona_phase",
        ] = 0  # First Lockdown
        data.loc[
            (data.index >= easing_start) & (data.index <= easing_end), "corona_phase"
        ] = 5  # Easing
        data.loc[
            (data.index >= lockdown2_start) & (data.index <= lockdown2_end),
            "corona_phase",
        ] = 1  # Second Lockdown
        data.loc[
            (data.index >= vaccionation_start) & (data.index <= vaccionation_end),
            "corona_phase",
        ] = 2  # Vaccination Rollout
        data.loc[(data.index > vaccionation_end), "corona_phase"] = (
            3  # Post-Vaccination
        )

        # Create rolling features
        data["rolling_mean_7"] = data["bike_count"].rolling(window=7).mean()
        data["rolling_mean_30"] = data["bike_count"].rolling(window=30).mean()
        data["rolling_std_7"] = data["bike_count"].rolling(window=7).std()
        data["rolling_std_30"] = data["bike_count"].rolling(window=30).std()

        data["diff_prev_day"] = data["bike_count"].diff()

        return data

    def create_features_iterative(self, data):
        """Create the features based on the data iteratively.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = data.copy()

        # Create time features
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["quarter"] = data.index.quarter
        data["weekday"] = data.index.weekday

        # Create lag features
        target_map = data["bike_count"].to_dict()
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 20, 28, 50]:
            data[f"lag_{lag}"] = (data.index - pd.Timedelta(days=lag)).map(target_map)

        for lag in [1, 2, 3, 4]:
            data[f"diff_lag_{lag}"] = data["bike_count"].diff(lag)

        # Create holiday feature
        data["is_holiday"] = 0
        bw_holidays = holidays.country_holidays("DE", subdiv="BW")
        for i, date in enumerate(data.index):
            if date in bw_holidays:
                data.at[date, "is_holiday"] = 1

        # Create corona feature
        lockdown1_start = pd.Timestamp("2020-03-16", tz="UTC")
        lockdown1_end = pd.Timestamp("2020-05-11", tz="UTC")
        easing_start = pd.Timestamp("2020-06-01", tz="UTC")
        easing_end = pd.Timestamp("2020-10-30", tz="UTC")
        lockdown2_start = pd.Timestamp("2020-11-02", tz="UTC")
        lockdown2_end = pd.Timestamp("2020-02-14", tz="UTC")
        vaccionation_start = pd.Timestamp("2021-03-01", tz="UTC")
        vaccionation_end = pd.Timestamp("2021-06-30", tz="UTC")
        data.loc[(data.index < lockdown1_start), "corona_phase"] = 4  # Pre-Pandemic
        data.loc[
            (data.index >= lockdown1_start) & (data.index <= lockdown1_end),
            "corona_phase",
        ] = 0  # First Lockdown
        data.loc[
            (data.index >= easing_start) & (data.index <= easing_end), "corona_phase"
        ] = 5  # Easing
        data.loc[
            (data.index >= lockdown2_start) & (data.index <= lockdown2_end),
            "corona_phase",
        ] = 1  # Second Lockdown
        data.loc[
            (data.index >= vaccionation_start) & (data.index <= vaccionation_end),
            "corona_phase",
        ] = 2  # Vaccination Rollout
        data.loc[(data.index > vaccionation_end), "corona_phase"] = (
            3  # Post-Vaccination
        )

        # Create rolling features
        data["rolling_mean_7"] = data["bike_count"].rolling(window=7).mean()
        data["rolling_mean_30"] = data["bike_count"].rolling(window=30).mean()
        data["rolling_std_7"] = data["bike_count"].rolling(window=7).std()
        data["rolling_std_30"] = data["bike_count"].rolling(window=30).std()

        data["diff_prev_day"] = data["bike_count"].diff()

        return data


if __name__ == "__main__":
    data_preparer = DataPreparing()
    data_w_weather = data_preparer.get_bike_and_weather_data()
    data_w_weather = data_preparer.create_features(data_w_weather)
    print(data_w_weather)
