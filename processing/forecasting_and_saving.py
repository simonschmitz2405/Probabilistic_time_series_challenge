from processing.DataPreparing import DataPreparing

import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import numpy as np

class forecasting_and_saving:
    """Class to forecast and to save the forecast to a csv file.

    Attributes:
        None

    Functions:
        predict_future_non_iterativ: Predict the future using the non-iterative approach.
        predict_future_iterativ: Predict the future using the iterative approach.
        save_forecasts: Save the forecast to a csv file.
    """    
    ###########################
    # For bike data ###########
    ###########################
    def predict_future_iterativ_bike(self, data_past, data_future, models, FEATURES, TARGET, quantiles, validation = False):
        """Generate the forecast for the future using the iterative approach.

        Args:
            data_past (pd.Dataframe): Dataframe with the past data.
            data_future (pd.Dataframe): Dataframe with the future data.
            models (object): Trained model to predict the future.
            FEATURES (dict[str]): Dict of features to use for the prediction.
            quantiles (list): List of quantiles to predict.

        Returns:
            future (pd.Dataframe): Dataframe with the forecast for the future.
        """
        if validation is False:
            last_day = data_past.index.max()
            start_date = last_day + pd.Timedelta(days=1)
            last_forecast = (start_date + pd.Timedelta(days=6))
            # Create the index for the future data
            future = pd.date_range(start=start_date, end=last_forecast, freq='d')

            # Take the weather data for the future
            future = data_future.loc[data_future.index.isin(future)].copy()



        if validation is True:
             future = data_future.copy()

        for quantile in quantiles:            
            for i in range(len(future)):
                # Create the features for the current day using the training data for lagged features
                future_current = future.iloc[:i+1].copy()
                future_current = pd.concat([data_past, future_current])
                data_preparer = DataPreparing()
                future_current = data_preparer.create_features_iterative(future_current, TARGET)

                # Just take the last rows of the future_current dataframe
                future_current = future_current.iloc[-(i+1):]
                X_future_current = future_current[FEATURES[quantile]]

                predictions = models[f"model_{quantile}"].predict(X_future_current)
                future.loc[future_current.index[-1], f'pred_{quantile}'] = predictions[-1]

                # Store this prediction now as the true value for the next iteration in the data. take therefore the 
                if quantile == 0.5:
                    future.loc[future_current.index[-1], TARGET] = predictions[-1]

        return future
    
    # Function specifically for the MSTL model (since it needs a different data structure)
    def predict_future_mstl_bike(self, data_past, models, quantiles):
        last_day = data_past.index.max()
        start_date = last_day + pd.Timedelta(days=1)
        last_forecast = (start_date + pd.Timedelta(days=6))
        # Create the index for the future data
        future = pd.date_range(start=start_date, end=last_forecast, freq='d')
        future = pd.DataFrame(index=future) 

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data_past.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["bike_count"]
        data_mstl.drop(columns=["bike_count"], inplace=True)

        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        for quantile in quantiles:
            future[f'pred_{quantile}'] = None # Placeholder to store the predictions

        for quantile in quantiles:
            y_pred = models[f"model_{quantile}"].forecast(df=data_mstl, h=len(future), level=[50, 95])
            future[f'pred_{quantile}'] = y_pred[
                    quantile_dict[quantile]
                ].tolist()
            
        return future
    
    def save_forecasts_bike(self, future_w_features):
        """Save the forecast to a csv file.

        Args:
            future_w_features (pd.Dataframe): Output of the prediction function.
        """
        # future_w_features = data_and_future[data_and_future['isFuture'] == True].copy()
        horizon_date = [timestamp.strftime('%Y-%m-%d') for timestamp in future_w_features.index[-6:].tolist()]
        horizons = [1, 2, 3, 4, 5, 6]
        prediction_columns = ['pred_0.025', 'pred_0.25', 'pred_0.5', 'pred_0.75', 'pred_0.975']
        now = datetime.now()
        data_date = now.strftime('%Y-%m-%d')
        item = "bikes"
        path = Path(__file__).parent.parent
        output_file = path / "output" / "single" / f"{data_date}_{item}.csv"

        # Open file to write
        with open(output_file, "w") as file:
            # Write the header
            header = "forecast_date,target,horizon,q0.025,q0.25,q0.5,q0.75,q0.975\n"
            file.write(header)

            # Write each forecast line in the specified format
            for idx, horizon in enumerate(horizons):
                horizon_time = horizon_date[idx]
                # Fetch the predictions for the current horizon
                quantile_values = [
                    f"{future_w_features.loc[horizon_time, column]:.3f}" 
                    for column in prediction_columns
                ]

                # Format the line and write it
                line = f"{data_date},{item},{horizon} day," + ",".join(quantile_values) + "\n"
                file.write(line)

        print(f"Bike forecasts saved to {output_file}")
    

    ###########################
    # For energy data #########
    ###########################
    def predict_future_iterative_energy(self, data_past, data_future, models, features, TARGET, quantiles, validation = False):
        data_preparer = DataPreparing()

        if validation is False:
            last_hour = data_past.index.max()
            now = datetime.now(tz=data_past.index.tz)
            last_forecast = last_forecast = (now + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
            future = pd.date_range(start=last_hour, end=last_forecast, freq='h')

            # Take the weather data for the future
            future = data_future.loc[data_future.index.isin(future)].copy()

        if validation is True:
            future = data_future.copy()

        # Do it iteratively for the 0.5 quantile since this is the true value for the next iteration
        for i in range(len(future)):
            future_current = future.iloc[:i+1].copy()
            future_current = pd.concat([data_past, future_current])
            future_current = data_preparer.create_features_iterative(future_current, TARGET)

            # Just take the last rows of the future_current dataframe
            future_current = future_current.iloc[-(i+1):]
            X_future_current = future_current[features[0.5]]

            predictions = models["model_0.5"].predict(X_future_current)
            future.loc[future_current.index[-1], 'pred_0.5'] = predictions[-1]

            # Store this prediction now as the true value for the next iteration in the data
            future.loc[future_current.index[-1], TARGET] = predictions[-1]


        # Get the future with the features for the other quantiles
        past_and_future = pd.concat([data_past, future])
        past_and_future = data_preparer.create_features_iterative(past_and_future, TARGET)
        # Just take the future data
        future = past_and_future.iloc[len(data_past):]



        for quantile in [0.025, 0.25, 0.75, 0.975]:       
            future.loc[:, f"pred_{quantile}"] = models[f"model_{quantile}"].predict(future[features[quantile]])


        return future
    
    # Function specifically for the MSTL model (since it needs a different data structure)
    def predict_future_mstl_energy(self, data_past, models, quantiles):
        last_hour = data_past.index.max()
        last_forecast = (last_hour + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
        future = pd.date_range(start=last_hour, end=last_forecast, freq='h')
        future = pd.DataFrame(index=future) 

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data_past.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["consumption"]
        data_mstl.drop(columns=["consumption"], inplace=True)

        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        for quantile in quantiles:
            future[f'pred_{quantile}'] = None # Placeholder to store the predictions

        for quantile in quantiles:
            y_pred = models[f"model_{quantile}"].forecast(df=data_mstl, h=len(future), level=[50, 95])

            future[f'pred_{quantile}'] = y_pred[
                    quantile_dict[quantile]
                ].tolist()
            
        return future

    def save_forecasts_energy(self, future_w_features):
        """Save the forecast to a csv file.

        Args:
            future_w_features (pd.Dataframe): Output of the prediction function.
        """
        now = datetime.now(tz=future_w_features.index.tz)
        horizon_date = [
            (now + pd.Timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0),
            (now + pd.Timedelta(days=2)).replace(hour=16, minute=0, second=0, microsecond=0),
            (now + pd.Timedelta(days=2)).replace(hour=20, minute=0, second=0, microsecond=0),
            (now + pd.Timedelta(days=3)).replace(hour=12, minute=0, second=0, microsecond=0),
            (now + pd.Timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0),
            (now + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
        ]
        horizons = [36, 40, 44, 60, 64, 68]
        prediction_columns = ['pred_0.025', 'pred_0.25', 'pred_0.5', 'pred_0.75', 'pred_0.975']
        data_date = now.strftime('%Y-%m-%d')
        item = "energy"
        path = Path(__file__).parent.parent
        output_file = path / "output" / "single" / f"{data_date}_{item}.csv"

        # Open file to write
        with open(output_file, "w") as file:
            # Write the header
            header = "forecast_date,target,horizon,q0.025,q0.25,q0.5,q0.75,q0.975\n"
            file.write(header)

            # Write each forecast line in the specified format
            for idx, horizon in enumerate(horizons):
                horizon_time = horizon_date[idx]
                # Fetch the predictions for the current horizon
                quantile_values = [
                    f"{future_w_features.loc[horizon_time, column]:.3f}" 
                    for column in prediction_columns
                ]

                # Format the line and write it
                line = f"{data_date},{item},{horizon} hour," + ",".join(quantile_values) + "\n"
                file.write(line)

        print(f"Energy forecasts saved to {output_file}")

    def predict_ensemble_model_bike(self, data_past, data_future, model_lightgbm, model_linearquantileregression, model_gradientboostingregressor, features_lightgbm, features_linearquantileregression, features_gradientboostingregressor, TARGET, quantiles, validation):
        """
        Train the ensemble model using the predictions from the quantile regression, lightgbm, gradient boosting and MSTL models.

        Args:
            data (pd.DataFrame): The input data
            TARGET (str): The target column
            quantiles (list): The quantiles to calculate the pinball loss

        Returns:
            results (dict): The results of the ensemble model
        """
        if validation is False:
            last_day = data_past.index.max()
            start_date = last_day + pd.Timedelta(days=1)
            last_forecast = (start_date + pd.Timedelta(days=6))
            # Create the index for the future data
            future = pd.date_range(start=start_date, end=last_forecast, freq='d')

            # Take the weather data for the future
            future = data_future.loc[data_future.index.isin(future)].copy()

        if validation is True:
                future = data_future.copy()

        for quantile in quantiles:            
            for i in range(len(future)):
                # Create the features for the current day using the training data for lagged features
                future_current = future.iloc[:i+1].copy()
                future_current = pd.concat([data_past, future_current])
                data_preparer = DataPreparing()
                future_current = data_preparer.create_features_iterative(future_current, TARGET)

                # Just take the last rows of the future_current dataframe
                future_current = future_current.iloc[-(i+1):]

                predictions = []

                if model_lightgbm is not None:
                    X_future_current = future_current[features_lightgbm[quantile]]
                    predictions_lightgbm = model_lightgbm[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_lightgbm[-1])

                if model_linearquantileregression is not None:
                    X_future_current = future_current[features_linearquantileregression[quantile]]
                    predictions_linearquantileregression = model_linearquantileregression[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_linearquantileregression[-1])

                if model_gradientboostingregressor is not None:
                    X_future_current = future_current[features_gradientboostingregressor[quantile]]
                    predictions_gradientboostingregressor = model_gradientboostingregressor[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_gradientboostingregressor[-1])
                
                # Calculate the mean prediction based on the three models and make sure that the mean is just taken
                # over the predicitons with are actually given. for example if there is none for the model then the mean 
                # should be taken only over the other two models
                future.loc[future_current.index[-1], f'pred_{quantile}'] = np.mean(predictions)


                # Store this prediction now as the true value for the next iteration in the data. take therefore the 
                if quantile == 0.5:
                    future.loc[future_current.index[-1], TARGET] = np.mean(predictions)

        return future
    

    def predict_ensemble_model_energy(self, data_past, data_future, model_lightgbm, model_linearquantileregression, model_gradientboostingregressor, features_lightgbm, features_linearquantileregression, features_gradientboostingregressor, TARGET, quantiles, validation):
        """
        Train the ensemble model using the predictions from the quantile regression, lightgbm, gradient boosting and MSTL models.

        Args:
            data (pd.DataFrame): The input data
            TARGET (str): The target column
            quantiles (list): The quantiles to calculate the pinball loss

        Returns:
            results (dict): The results of the ensemble model
        """
        if validation is False:
            last_hour = data_past.index.max()
            now = datetime.now(tz=data_past.index.tz)
            last_forecast = last_forecast = (now + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
            future = pd.date_range(start=last_hour, end=last_forecast, freq='h')

            # Take the weather data for the future
            future = data_future.loc[data_future.index.isin(future)].copy()

        if validation is True:
             future = data_future.copy()

        for quantile in quantiles:            
            for i in range(len(future)):
                # Create the features for the current day using the training data for lagged features
                future_current = future.iloc[:i+1].copy()
                future_current = pd.concat([data_past, future_current])
                data_preparer = DataPreparing()
                future_current = data_preparer.create_features_iterative(future_current, TARGET)

                # Just take the last rows of the future_current dataframe
                future_current = future_current.iloc[-(i+1):]

                predictions = []

                if model_lightgbm is not None:
                    X_future_current = future_current[features_lightgbm[quantile]]
                    predictions_lightgbm = model_lightgbm[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_lightgbm[-1])

                if model_linearquantileregression is not None:
                    X_future_current = future_current[features_linearquantileregression[quantile]]
                    predictions_linearquantileregression = model_linearquantileregression[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_linearquantileregression[-1])

                if model_gradientboostingregressor is not None:
                    X_future_current = future_current[features_gradientboostingregressor[quantile]]
                    predictions_gradientboostingregressor = model_gradientboostingregressor[f"model_{quantile}"].predict(X_future_current)
                    predictions.append(predictions_gradientboostingregressor[-1])
                
                # Calculate the mean prediction based on the three models and make sure that the mean is just taken
                # over the predicitons with are actually given. for example if there is none for the model then the mean 
                # should be taken only over the other two models
                future.loc[future_current.index[-1], f'pred_{quantile}'] = np.mean(predictions)


                # Store this prediction now as the true value for the next iteration in the data. take therefore the 
                if quantile == 0.5:
                    future.loc[future_current.index[-1], TARGET] = np.mean(predictions)

        return future