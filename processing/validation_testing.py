from processing.DataPreparing import DataPreparing
from processing.forecasting_and_saving import forecasting_and_saving

from models.LightGBM_Model import LightGBM_Model
from models.MSTL_Model import MSTL_Model
from models.LinearQuantileRegression_Model import LinearQuantileRegression_Model
from models.GradientBoostingRegressor_Model import GradientBoostingRegressor_Model
from models.Baseline_Model import Baseline_Model

from hyperparameter_optuna import hyperparameter_optuna
import optuna

from matplotlib import pyplot as plt
import os
from pathlib import Path
from typing import Tuple, Dict, List
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import d2_pinball_score
from statsmodels.tsa.seasonal import MSTL
from statsforecast import StatsForecast
from statsforecast.models import MSTL

# Add here to give back per iteration the test_and_train_data and the validation_data to plot it in the notebook later
class validation_testing:
    def interval_score_metric(self, lower_bound, upper_bound, y_true, alpha=0.95):
        score = (upper_bound - lower_bound) + 2 / alpha * (lower_bound - y_true) * (y_true < lower_bound) + 2 / alpha * (y_true - upper_bound) * (y_true > upper_bound)
        return score.mean()


    def testing_on_validation_set_bike(self,data, iterations, selected_model, hyperparameters) -> Dict[str, float]:   
        result_validation = {}
        data_dict = {}
        future_dict = {}

        data = data.copy()
        data.loc[:,'bike_count'] = data['bike_count'].fillna(int(data['bike_count'].mean())) 

        validation_end = '01-09-2025' # Validation End for first iteration 
        validation_end = pd.to_datetime(validation_end).tz_localize('UTC')

        pinnball_loss_each_iteration = []
        d2_pinball_loss_each_iteration = []
        interval_score_50_percent_each_iteration = []
        interval_score_95_percent_each_iteration = []
        calibration_each_iteration = {quantile: [] for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        sharpness_each_iteration = []


        for i in range(iterations):
            validation_start = pd.to_datetime(validation_end) - pd.Timedelta(days=7) # 7 days before the validation end
            test_and_train_data = data.loc[(data.index < validation_start)].copy()
            validation_data = data.loc[(data.index >= validation_start) & (data.index < validation_end)].copy()
            data_dict[f"Train/Test_{i+1}"] = test_and_train_data
            data_dict[f"Validation_{i+1}"] = validation_data

            features = None
            model = None
            future = None

            if selected_model == "lightgbm":
                learning_rate_lgbm = hyperparameters["bike"]["lightgbm"]["learning_rate"]
                n_estimators_lgbm = hyperparameters["bike"]["lightgbm"]["n_estimators"]
                num_leaves_lgbm = hyperparameters["bike"]["lightgbm"]["num_leaves"]
                reg_alpha_lgbm = hyperparameters["bike"]["lightgbm"]["reg_alpha"]
                features = hyperparameters["bike"]["lightgbm"]["features"]
                # Train model
                lgbm = LightGBM_Model() 
                # model, pred_lgbm, res_lgbm, y_true_dict = lgbm.lightgbm_model(test_and_train_data, features, TARGET_bike, learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975], 1, 0) # Changed from 5 , 30 to 1, 0
                model = lgbm.lightgbm_model_final(test_and_train_data, features, 'bike_count', learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975]) # Changed from 5 , 30 to 1, 0

                # Predict the future iteratively
                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_future_iterativ_bike(test_and_train_data, validation_data, model, features, 'bike_count', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == "linearquantileregression":
                fit_intercept_lqr = hyperparameters["bike"]["linearquantileregression"]["fit_intercept"]
                alpha_lqr = hyperparameters["bike"]["linearquantileregression"]["alpha"]
                features = hyperparameters["bike"]["linearquantileregression"]["features"]
                # Train model
                lqr = LinearQuantileRegression_Model()
                model = lqr.quantile_regression_model_final(test_and_train_data, features, 'bike_count', fit_intercept_lqr[0.025], alpha_lqr, [0.025, 0.25, 0.5, 0.75, 0.975])
                
                # Predict the future iteratively
                forecaster_and_saver = forecasting_and_saving()
                validation_data = validation_data.fillna(validation_data.mean()) # Same weather data are nan therefore fill with mean
                future = forecaster_and_saver.predict_future_iterativ_bike(test_and_train_data, validation_data, model, features, 'bike_count', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == "gradientboostingregressor":
                learning_rate_gb = hyperparameters["bike"]["gradientboostingregressor"]["learning_rate"]
                n_estimators_gb = hyperparameters["bike"]["gradientboostingregressor"]["n_estimators"]
                features = hyperparameters["bike"]["gradientboostingregressor"]["features"]

                gbr = GradientBoostingRegressor_Model()
                model = gbr.gradientboostingregressor_model_final(test_and_train_data, features, 'bike_count', learning_rate_gb, n_estimators_gb,[0.025, 0.25, 0.5, 0.75, 0.975])

                # Predict the future iteratively
                forecaster_and_saver = forecasting_and_saving()
                validation_data = validation_data.fillna(validation_data.mean()) # Same weather data are nan therefore fill with mean
                future = forecaster_and_saver.predict_future_iterativ_bike(test_and_train_data, validation_data, model, features, 'bike_count', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == "mstl":
                season_length_one = hyperparameters["bike"]["mstl"]["season_length_one"]
                season_length_two = hyperparameters["bike"]["mstl"]["season_length_two"]
                historical = hyperparameters["bike"]["mstl"]["historical_length"]
                mstl = MSTL_Model()
                model = mstl.mstl_model_bike_final(test_and_train_data, historical, season_length_one, season_length_two, [0.025, 0.25, 0.5, 0.75, 0.975])

                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_future_mstl_bike(test_and_train_data, model, [0.025, 0.25, 0.5, 0.75, 0.975])
                future_dict[f"Future_{i+1}"] = future
                

            if selected_model == "ensemble":
                # Overall Pinball loss: 59.931859667636594
                # Overall D2 Pinball score: 0.49257815601581595
                learning_rate_lgbm = hyperparameters["bike"]["lightgbm"]["learning_rate"]
                n_estimators_lgbm = hyperparameters["bike"]["lightgbm"]["n_estimators"]
                num_leaves_lgbm = hyperparameters["bike"]["lightgbm"]["num_leaves"]
                reg_alpha_lgbm = hyperparameters["bike"]["lightgbm"]["reg_alpha"]
                features_lightgbm_bike = hyperparameters["bike"]["lightgbm"]["features"]
                lgbm = LightGBM_Model()
                model_lgbm_bike = lgbm.lightgbm_model_final(test_and_train_data, features_lightgbm_bike, 'bike_count', learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975])


                fit_intercept_lqr = hyperparameters["bike"]["linearquantileregression"]["fit_intercept"]
                alpha_lqr = hyperparameters["bike"]["linearquantileregression"]["alpha"]
                features_linearquantileregression_bike = hyperparameters["bike"]["linearquantileregression"]["features"]
                lqr = LinearQuantileRegression_Model()
                model_lqr_bike = lqr.quantile_regression_model_final(test_and_train_data, features_linearquantileregression_bike, 'bike_count', fit_intercept_lqr[0.025], alpha_lqr, [0.025, 0.25, 0.5, 0.75, 0.975])
                
                learning_rate_gb = hyperparameters["bike"]["gradientboostingregressor"]["learning_rate"]
                n_estimators_gb = hyperparameters["bike"]["gradientboostingregressor"]["n_estimators"]
                features_gb_bike = hyperparameters["bike"]["gradientboostingregressor"]["features"]
                gbr = GradientBoostingRegressor_Model()
                model_gbr_bike = gbr.gradientboostingregressor_model_final(test_and_train_data, features_gb_bike, 'bike_count', learning_rate_gb, n_estimators_gb, [0.025, 0.25, 0.5, 0.75, 0.975])
                
                validation_data = validation_data.fillna(validation_data.mean())
                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_ensemble_model_bike(test_and_train_data, validation_data, model_lgbm_bike, model_lqr_bike, model_gbr_bike, features_lightgbm_bike, features_linearquantileregression_bike, features_gb_bike, 'bike_count', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == 'baseline':
                last_day = test_and_train_data.index.max()
                start_date = last_day + pd.Timedelta(days=1)
                last_forecast = (start_date + pd.Timedelta(days=6))
               
                # Create the index for the future data
                future = pd.date_range(start=start_date, end=last_forecast, freq='d')

                # Take the weather data for the future
                future = validation_data.loc[validation_data.index.isin(future)].copy()

                for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                    for date in validation_data.index:
                        weekday = date.weekday()

                        same_weekday_data = test_and_train_data.loc[test_and_train_data.index.weekday == weekday]
                        last_100_days = same_weekday_data.iloc[-100:]
                        quantile_predictions = last_100_days['bike_count'].quantile(quantile)
                        future.loc[date, f"pred_{quantile}"] = quantile_predictions


            # Store the results
            pinnball_loss_each_quantile = []
            d2_pinball_loss_each_quantile = []
            sharpness_for_iteration = []

            # Calculate the evaluation metrics for each quantile
            for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                pinnball_loss = mean_pinball_loss(validation_data["bike_count"], future[f"pred_{quantile}"], alpha=quantile)
                pinnball_loss_each_quantile.append(pinnball_loss)
                d2_pinball = d2_pinball_score(validation_data["bike_count"], future[f"pred_{quantile}"], alpha=quantile)
                d2_pinball_loss_each_quantile.append(d2_pinball)
                observed_proportion = np.mean(validation_data["bike_count"] <= future[f"pred_{quantile}"])
                calibration_error = abs(observed_proportion - quantile)
                calibration_each_iteration[quantile].append(calibration_error)
                result_validation[f"Calibration for iteration {i} for quantile {quantile}"] = calibration_error
                result_validation[f"Pinball loss for iteration {i} and quantile {quantile}"] = pinnball_loss
                result_validation[f"D2 Pinball score for iteration {i} and quantile {quantile}"] = d2_pinball


            result_validation[f"Pinball loss for iteration {i}"] = np.mean(pinnball_loss_each_quantile)
            result_validation[f"D2 Pinball score for iteration {i}"] = np.mean(d2_pinball_loss_each_quantile)
            

            
            # Calculate the interval score for 95% and 50% prediction interval
            interval_score_95_percent = self.interval_score_metric(future[f"pred_0.025"], future[f"pred_0.975"], validation_data["bike_count"], alpha=0.95)
            interval_score_50_percent = self.interval_score_metric(future[f"pred_0.25"], future[f"pred_0.75"], validation_data["bike_count"], alpha=0.5)

            
            result_validation[f"Interval Score 95% for iteration {i}"] = interval_score_95_percent
            result_validation[f"Interval Score 50% for iteration {i}"] = interval_score_50_percent

            pinnball_loss_each_iteration.append(np.mean(pinnball_loss_each_quantile))
            d2_pinball_loss_each_iteration.append(np.mean(d2_pinball_loss_each_quantile))
            interval_score_50_percent_each_iteration.append(interval_score_50_percent)
            interval_score_95_percent_each_iteration.append(interval_score_95_percent)

            validation_end = validation_start - pd.Timedelta(weeks = 3) # Set the new validation end to the start of the previous validation start

                # Sharpness: Compute interval width for middle intervals
            for lower, upper in [(0.025, 0.975), (0.25, 0.75)]:
                interval_width = future[f"pred_{upper}"] - future[f"pred_{lower}"]
                sharpness_for_iteration.append(interval_width.mean())
        
            sharpness_each_iteration.append(np.mean(sharpness_for_iteration))
            result_validation[f"Sharpness for iteration {i}"] = np.mean(sharpness_for_iteration)

        overall_pinball_loss = np.mean(pinnball_loss_each_iteration)
        overall_d2_pinball_score = np.mean(d2_pinball_loss_each_iteration)
        interval_score_95_percent = np.mean(interval_score_95_percent_each_iteration)
        interval_score_50_percent = np.mean(interval_score_50_percent_each_iteration)

        for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
            result_validation[f"Overall Pinball loss for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "Pinball loss for iteration" in key]
            )
            result_validation[f"Overall D2 Pinball score for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "D2 Pinball score for iteration" in key]
            )

        overall_calibration = {quantile: np.mean(calibration_each_iteration[quantile]) for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        overall_sharpness = np.mean(sharpness_each_iteration)


        result_validation["Overall Pinball loss"] = overall_pinball_loss
        result_validation["Overall D2 Pinball score"] = overall_d2_pinball_score
        result_validation["Interval Score 95%"] = interval_score_95_percent
        result_validation["Interval Score 50%"] = interval_score_50_percent
        result_validation["Overall Calibration"] = overall_calibration
        result_validation["Overall Sharpness"] = overall_sharpness


        print(f"Overall Pinball loss: {overall_pinball_loss}")
        print(f"Overall D2 Pinball score: {overall_d2_pinball_score}")
        print(f"Interval Score 95%: {interval_score_95_percent}")
        print(f"Interval Score 50%: {interval_score_50_percent}")
        print(f"Overall Calibration: {overall_calibration}")
        print(f"Overall Sharpness: {overall_sharpness}")


        return result_validation, future_dict, data_dict
    

    def testing_on_validation_set_energy(self, data_energy, iterations, selected_model, hyperparameters) -> Dict[str, float]:   
        result_validation = {}
        data_dict = {}
        future_dict = {}

        data_energy = data_energy.copy()

        validation_end = '01-04-2025' # Validation End for first iteration 
        validation_end = pd.to_datetime(validation_end).tz_localize('UTC')

        pinnball_loss_each_iteration = []
        d2_pinball_loss_each_iteration = []
        interval_score_50_percent_each_iteration = []
        interval_score_95_percent_each_iteration = []
        calibration_each_iteration = {quantile: [] for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        sharpness_each_iteration = []


        for i in range(iterations):
            # Data Importing
            data = data_energy.copy()
            validation_start = pd.to_datetime(validation_end) - pd.Timedelta(days=3) # 3 days of prediction time
            test_and_train_data = data.loc[(data.index < validation_start)].copy()
            validation_data = data.loc[(data.index >= validation_start) & (data.index < validation_end)].copy()
            data_dict[f"Train/Test_{i+1}"] = test_and_train_data
            data_dict[f"Validation_{i+1}"] = validation_data

            features = None
            model = None
            future = None

            if selected_model == "lightgbm":
                learning_rate_lgbm = hyperparameters["energy"]["lightgbm"]["learning_rate"]
                n_estimators_lgbm = hyperparameters["energy"]["lightgbm"]["n_estimators"]
                num_leaves_lgbm = hyperparameters["energy"]["lightgbm"]["num_leaves"]
                reg_alpha_lgbm = hyperparameters["energy"]["lightgbm"]["reg_alpha"]
                features = hyperparameters["energy"]["lightgbm"]["features"]
                # Train model
                lgbm = LightGBM_Model() 
                model = lgbm.lightgbm_model_final(test_and_train_data, features, 'consumption', learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975])
                # Predict the future iteratively
                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_future_iterative_energy(test_and_train_data, validation_data, model, features, 'consumption', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == "linearquantileregression":
                fit_intercept_lqr = hyperparameters["energy"]["linearquantileregression"]["fit_intercept"]
                alpha_lqr = hyperparameters["energy"]["linearquantileregression"]["alpha"]
                features = hyperparameters["energy"]["lightgbm"]["features"] # Changed to lightgbm features
                # Train model
                test_and_train_data = test_and_train_data.fillna(test_and_train_data.mean()) 
                lqr = LinearQuantileRegression_Model()
                model= lqr.quantile_regression_model_final(test_and_train_data, features, 'consumption', fit_intercept_lqr[0.025], alpha_lqr, [0.025, 0.25, 0.5, 0.75, 0.975])
                
                # Predict the future iteratively
                forecaster_and_saver = forecasting_and_saving()
                validation_data = validation_data.fillna(validation_data.mean()) # Same weather data are nan therefore fill with mean
                future = forecaster_and_saver.predict_future_iterative_energy(test_and_train_data, validation_data, model, features, 'consumption', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            # if selected_model == "gradientboostingregressor":
            #     features = features_gradientboostingregressor_energy
            #     test_and_train_data = test_and_train_data.fillna(test_and_train_data.mean()) 
            #     gbr = GradientBoostingRegressor_Model()
            #     model, pred_gbr_energy, res_gbr_energy = gbr.gradientboostingregressor_model(test_and_train_data, features, TARGET_energy, [0.025, 0.25, 0.5, 0.75, 0.975], 5, 72*5)

            #     # Predict the future iteratively
            #     forecaster_and_saver = forecasting_and_saving()
            #     validation_data = validation_data.fillna(validation_data.mean()) # Same weather data are nan therefore fill with mean
            #     future = forecaster_and_saver.predict_future_iterative_energy(test_and_train_data, validation_data, model, features, TARGET_energy, [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
            #     future_dict[f"Future_{i+1}"] = future


            if selected_model == "mstl":
                season_length_one = hyperparameters["energy"]["mstl"]["season_length_one"]
                season_length_two = hyperparameters["energy"]["mstl"]["season_length_two"]
                season_length_three = hyperparameters["energy"]["mstl"]["season_length_three"]
                historical = hyperparameters["energy"]["mstl"]["historical_length"]
                mstl = MSTL_Model()
                model = mstl.mstl_model_energy_final(test_and_train_data, historical, season_length_one, season_length_two, season_length_three, [0.025, 0.25, 0.5, 0.75, 0.975])

                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_future_mstl_bike(test_and_train_data, model, [0.025, 0.25, 0.5, 0.75, 0.975])
                future_dict[f"Future_{i+1}"] = future

            # if selected_model == "ensemble":
            #     test_and_train_data = test_and_train_data.fillna(test_and_train_data.mean()) 

            #     lgbm = LightGBM_Model()
            #     model_lgbm_energy = lgbm.lightgbm_model_final(test_and_train_data, features_lightgbm_energy, TARGET_energy, learning_rate_lightgbm, n_estimators, num_leaves, reg_alpha, [0.025, 0.25, 0.5, 0.75, 0.975])

            #     lqr = LinearQuantileRegression_Model()
            #     model_lqr_energy  = lqr.quantile_regression_model_final(test_and_train_data, features_linearquantileregression_energy, TARGET_energy, fit_intercept[0.025], alpha, [0.025, 0.25, 0.5, 0.75, 0.975])
                
            #     # gbr = GradientBoostingRegressor_Model()
            #     # model_gbr_energy = gbr.gradientboostingregressor_model_final(test_and_train_data, features_gradientboostingregressor_energy, TARGET_energy, [0.025, 0.25, 0.5, 0.75, 0.975])
                
            #     validation_data = validation_data.fillna(validation_data.mean())
            #     forecaster_and_saver = forecasting_and_saving()
            #     future = forecaster_and_saver.predict_ensemble_model_energy(test_and_train_data, validation_data, model_lgbm_energy, model_lqr_energy, model_gbr_energy, features_lightgbm_energy, features_linearquantileregression_energy, features_gradientboostingregressor_energy, TARGET_energy, [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
            #     future_dict[f"Future_{i+1}"] = future


            if selected_model == 'baseline':
                last_hour = test_and_train_data.index.max()
                last_forecast = (last_hour + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
                start_date = last_hour + pd.Timedelta(hours=1)
                future = pd.date_range(start=start_date, end=last_forecast, freq='h')
               
                # Take the weather data for the future
                future = validation_data.loc[validation_data.index.isin(future)].copy()

                for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                    for date in validation_data.index:
                        hour = date.hour

                        same_hour_data = test_and_train_data.loc[test_and_train_data.index.hour == hour]
                        last_100_days = same_hour_data.iloc[-100:]
                        quantile_predictions = last_100_days['consumption'].quantile(quantile)
                        future.loc[date, f"pred_{quantile}"] = quantile_predictions


            pinnball_loss_each_quantile = []
            d2_pinball_loss_each_quantile = []
            sharpness_for_iteration = []


            for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                # Select the 6 hours which have to be predicted
                # horizon_date = [
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=12)),
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=16)), 
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=20)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=12)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=16)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=20))
                # ]
                # Select the 6 hours which have to be predicted from future and the true values from the validation data
                # future_horizon = future.loc[horizon_date]
                # true_values = validation_data.loc[horizon_date]

                horizon_end = validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=20)
                # Select the range of hours from validation_start to horizon_end
                future = future.loc[validation_start:horizon_end].copy()

                # Select the corresponding true values from the validation data
                validation_data = validation_data.loc[validation_start:horizon_end].copy()

            

                pinnball_loss = mean_pinball_loss(validation_data["consumption"], future[f"pred_{quantile}"], alpha=quantile)
                pinnball_loss_each_quantile.append(pinnball_loss)
                d2_pinball = d2_pinball_score(validation_data["consumption"], future[f"pred_{quantile}"], alpha=quantile)
                d2_pinball_loss_each_quantile.append(d2_pinball)
                observed_proportion = np.mean(validation_data["consumption"] <= future[f"pred_{quantile}"])
                calibration_error = abs(observed_proportion - quantile)
                calibration_each_iteration[quantile].append(calibration_error)

                result_validation[f"Calibration for iteration {i} for quantile {quantile}"] = calibration_error
                result_validation[f"Pinball loss for iteration {i} and quantile {quantile}"] = pinnball_loss
                result_validation[f"D2 Pinball score for iteration {i} and quantile {quantile}"] = d2_pinball

            result_validation[f"Pinball loss for iteration {i}"] = np.mean(pinnball_loss_each_quantile)
            result_validation[f"D2 Pinball score for iteration {i}"] = np.mean(d2_pinball_loss_each_quantile)

            # Calculate the interval score for 95% and 50% prediction interval
            interval_score_95_percent = self.interval_score_metric(future[f"pred_0.025"], future[f"pred_0.975"], validation_data["consumption"], alpha=0.95)
            interval_score_50_percent = self.interval_score_metric(future[f"pred_0.25"], future[f"pred_0.75"], validation_data["consumption"], alpha=0.5)

            result_validation[f"Interval Score 95% for iteration {i}"] = interval_score_95_percent
            result_validation[f"Interval Score 50% for iteration {i}"] = interval_score_50_percent


            print(f"Pinball loss for Validation from {validation_start} to {validation_end}: {np.mean(pinnball_loss_each_quantile)}")
            print(f"D2 Pinball score for Validation from {validation_start} to {validation_end}: {np.mean(d2_pinball_loss_each_quantile)}")

            pinnball_loss_each_iteration.append(np.mean(pinnball_loss_each_quantile))
            d2_pinball_loss_each_iteration.append(np.mean(d2_pinball_loss_each_quantile))
            interval_score_50_percent_each_iteration.append(interval_score_50_percent)
            interval_score_95_percent_each_iteration.append(interval_score_95_percent)
            
            validation_end = validation_start - pd.Timedelta(weeks = 1) # Set the new validation end to the start of the previous validation start

            # Sharpness: Compute interval width for middle intervals
            for lower, upper in [(0.025, 0.975), (0.25, 0.75)]:
                interval_width = future[f"pred_{upper}"] - future[f"pred_{lower}"]
                sharpness_for_iteration.append(interval_width.mean())
        
            sharpness_each_iteration.append(np.mean(sharpness_for_iteration))
            result_validation[f"Sharpness for iteration {i}"] = np.mean(sharpness_for_iteration)

        overall_pinball_loss = np.mean([result_validation[key] for key in result_validation.keys() if "Pinball" in key])
        overall_d2_pinball_score = np.mean([result_validation[key] for key in result_validation.keys() if "D2" in key])
        interval_score_95_percent = np.mean(interval_score_95_percent_each_iteration)
        interval_score_50_percent = np.mean(interval_score_50_percent_each_iteration)


        for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
            result_validation[f"Overall Pinball loss for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "Pinball loss for iteration" in key]
            )
            result_validation[f"Overall D2 Pinball score for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "D2 Pinball score for iteration" in key]
            )

        overall_calibration = {quantile: np.mean(calibration_each_iteration[quantile]) for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        overall_sharpness = np.mean(sharpness_each_iteration)

        result_validation["Overall Pinball loss"] = overall_pinball_loss
        result_validation["Overall D2 Pinball score"] = overall_d2_pinball_score
        result_validation["Interval Score 95%"] = interval_score_95_percent
        result_validation["Interval Score 50%"] = interval_score_50_percent
        result_validation["Overall Calibration"] = overall_calibration
        result_validation["Overall Sharpness"] = overall_sharpness

        print(f"Overall Pinball loss: {overall_pinball_loss}")
        print(f"Overall D2 Pinball score: {overall_d2_pinball_score}")
        print(f"Interval Score 95%: {interval_score_95_percent}")
        print(f"Interval Score 50%: {interval_score_50_percent}")
        print(f"Overall Calibration: {overall_calibration}")
        print(f"Overall Sharpness: {overall_sharpness}")


        return result_validation, future_dict, data_dict



    def testing_on_validation_set_energy_test(self, data_energy, iterations, selected_model, hyperparameters) -> Dict[str, float]:   
        result_validation = {}
        data_dict = {}
        future_dict = {}

        data_energy = data_energy.copy()

        validation_start = pd.Timestamp("2024-10-29 00:00:00", tz="UTC")
        data = data_energy.copy()
        validation_end = pd.to_datetime(validation_start) + pd.Timedelta(days = 3) # Validation End for first iteration
        test_and_train_data = data.loc[(data.index < validation_start)].copy()
        validation_data = data.loc[(data.index >= validation_start) & (data.index < validation_end)].copy()


        features = None
        model = None
        future = None


        if selected_model == "lightgbm":
                learning_rate_lgbm = hyperparameters["energy"]["lightgbm"]["learning_rate"]
                n_estimators_lgbm = hyperparameters["energy"]["lightgbm"]["n_estimators"]
                num_leaves_lgbm = hyperparameters["energy"]["lightgbm"]["num_leaves"]
                reg_alpha_lgbm = hyperparameters["energy"]["lightgbm"]["reg_alpha"]
                features = hyperparameters["energy"]["lightgbm"]["features"]
                # Train model
                lgbm = LightGBM_Model() 
                model = lgbm.lightgbm_model_final(test_and_train_data, features, 'consumption', learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975])

        if selected_model == "linearquantileregression":
            fit_intercept_lqr = hyperparameters["energy"]["linearquantileregression"]["fit_intercept"]
            alpha_lqr = hyperparameters["energy"]["linearquantileregression"]["alpha"]
            features = hyperparameters['energy']['features_for_optuna'] # Changed to all features
            features = {
                0.025: features,
                0.25: features,
                0.5: features,
                0.75: features,
                0.975: features
            }
            # Train model
            test_and_train_data = test_and_train_data.fillna(test_and_train_data.mean()) 
            lqr = LinearQuantileRegression_Model()
            model= lqr.quantile_regression_model_final(test_and_train_data, features, 'consumption', fit_intercept_lqr[0.025], alpha_lqr, [0.025, 0.25, 0.5, 0.75, 0.975])


        if selected_model == "mstl":
            season_length_one = hyperparameters["energy"]["mstl"]["season_length_one"]
            season_length_two = hyperparameters["energy"]["mstl"]["season_length_two"]
            historical = hyperparameters["energy"]["mstl"]["historical_length"]
            mstl = MSTL_Model()
            model = mstl.mstl_model_energy_final(test_and_train_data, historical, season_length_one, season_length_two, [0.025, 0.25, 0.5, 0.75, 0.975])


        if selected_model == "ensemble":
            test_and_train_data = test_and_train_data.fillna(test_and_train_data.mean()) 

            learning_rate_lgbm = hyperparameters["energy"]["lightgbm"]["learning_rate"]
            n_estimators_lgbm = hyperparameters["energy"]["lightgbm"]["n_estimators"]
            num_leaves_lgbm = hyperparameters["energy"]["lightgbm"]["num_leaves"]
            reg_alpha_lgbm = hyperparameters["energy"]["lightgbm"]["reg_alpha"]
            features_gbm = hyperparameters["energy"]["lightgbm"]["features"]
            lgbm = LightGBM_Model()
            model_lgbm_energy = lgbm.lightgbm_model_final(test_and_train_data, features_gbm, 'consumption', learning_rate_lgbm, n_estimators_lgbm, num_leaves_lgbm, reg_alpha_lgbm, [0.025, 0.25, 0.5, 0.75, 0.975])

            lqr = LinearQuantileRegression_Model()
            fit_intercept_lqr = hyperparameters["energy"]["linearquantileregression"]["fit_intercept"]
            alpha_lqr = hyperparameters["energy"]["linearquantileregression"]["alpha"]
            features_lqr = hyperparameters['energy']['features_for_optuna'] # Changed to all features
            features_lqr = {
                0.025: features_lqr,
                0.25: features_lqr,
                0.5: features_lqr,
                0.75: features_lqr,
                0.975: features_lqr
            }
            model_lqr_energy  = lqr.quantile_regression_model_final(test_and_train_data, features_lqr, 'consumption', fit_intercept_lqr[0.025], alpha_lqr, [0.025, 0.25, 0.5, 0.75, 0.975])

            


        pinnball_loss_each_iteration = []
        d2_pinball_loss_each_iteration = []
        interval_score_50_percent_each_iteration = []
        interval_score_95_percent_each_iteration = []
        calibration_each_iteration = {quantile: [] for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        sharpness_each_iteration = []

        for i in range(iterations):

            # Define the timesteps
            test_and_train_data = data.loc[(data.index < validation_start)].copy()

            validation_end = pd.to_datetime(validation_start) + pd.Timedelta(days= 3) # Validation End for first iteration
            validation_data = data.loc[(data.index >= validation_start) & (data.index < validation_end)].copy()

            data_dict[f"Train/Test_{i+1}"] = test_and_train_data
            data_dict[f"Validation_{i+1}"] = validation_data

            forecaster_and_saver = forecasting_and_saving()
            if selected_model == 'mstl':
                future = forecaster_and_saver.predict_future_mstl_energy(test_and_train_data, model, [0.025, 0.25, 0.5, 0.75, 0.975])
                future_dict[f"Future_{i+1}"] = future
            # Predict the future iteratively
            if selected_model == 'lightgbm' or selected_model == 'linearquantileregression':
                validation_data = validation_data.fillna(validation_data.mean()) # THIS WAS BEFORE JUST FOR LINEAR QUANTILE REGRESSION BUT NOW ALSO FOR LIGHTGBM
                future = forecaster_and_saver.predict_future_iterative_energy(test_and_train_data, validation_data, model, features, 'consumption', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                future_dict[f"Future_{i+1}"] = future

            if selected_model == 'baseline':
                last_hour = test_and_train_data.index.max()
                last_forecast = (last_hour + pd.Timedelta(days=3)).replace(hour=20, minute=0, second=0, microsecond=0)
                start_date = last_hour + pd.Timedelta(hours=1)
                future = pd.date_range(start=start_date, end=last_forecast, freq='h')
                
                # Take the weather data for the future
                future = validation_data.loc[validation_data.index.isin(future)].copy()

                for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                    for date in validation_data.index:
                        hour = date.hour

                        same_hour_data = test_and_train_data.loc[test_and_train_data.index.hour == hour]
                        last_100_days = same_hour_data.iloc[-100:]
                        quantile_predictions = last_100_days['consumption'].quantile(quantile)
                        future.loc[date, f"pred_{quantile}"] = quantile_predictions
                future_dict[f"Future_{i+1}"] = future

            if selected_model == "ensemble":
                print('when call baseline', validation_data)
                validation_data = validation_data.fillna(validation_data.mean())
                forecaster_and_saver = forecasting_and_saving()
                future = forecaster_and_saver.predict_ensemble_model_energy(test_and_train_data, validation_data, model_lgbm_energy, model_lqr_energy, None, features_gbm, features_lqr, None, 'consumption', [0.025, 0.25, 0.5, 0.75, 0.975], validation = True)
                print('After predict', future)
                future_dict[f"Future_{i+1}"] = future



            pinnball_loss_each_quantile = []
            d2_pinball_loss_each_quantile = []
            sharpness_for_iteration = []


            for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
                # Select the 6 hours which have to be predicted
                # horizon_date = [
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=12)),
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=16)), 
                #     (validation_start + pd.Timedelta(days=1) + pd.Timedelta(hours=20)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=12)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=16)),
                #     (validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=20))
                # ]
                # Select the 6 hours which have to be predicted from future and the true values from the validation data
                # future_horizon = future.loc[horizon_date]
                # true_values = validation_data.loc[horizon_date]

                horizon_end = validation_start + pd.Timedelta(days=2) + pd.Timedelta(hours=20)
                # Select the range of hours from validation_start to horizon_end
                future = future.loc[validation_start:horizon_end].copy()

                # Select the corresponding true values from the validation data
                validation_data = validation_data.loc[validation_start:horizon_end].copy()

            

                pinnball_loss = mean_pinball_loss(validation_data["consumption"], future[f"pred_{quantile}"], alpha=quantile)
                pinnball_loss_each_quantile.append(pinnball_loss)
                d2_pinball = d2_pinball_score(validation_data["consumption"], future[f"pred_{quantile}"], alpha=quantile)
                d2_pinball_loss_each_quantile.append(d2_pinball)
                observed_proportion = np.mean(validation_data["consumption"] <= future[f"pred_{quantile}"])
                calibration_error = abs(observed_proportion - quantile)
                calibration_each_iteration[quantile].append(calibration_error)

                result_validation[f"Calibration for iteration {i} for quantile {quantile}"] = calibration_error
                result_validation[f"Pinball loss for iteration {i} and quantile {quantile}"] = pinnball_loss
                result_validation[f"D2 Pinball score for iteration {i} and quantile {quantile}"] = d2_pinball

            result_validation[f"Pinball loss for iteration {i}"] = np.mean(pinnball_loss_each_quantile)
            result_validation[f"D2 Pinball score for iteration {i}"] = np.mean(d2_pinball_loss_each_quantile)

            # Calculate the interval score for 95% and 50% prediction interval
            interval_score_95_percent = self.interval_score_metric(future[f"pred_0.025"], future[f"pred_0.975"], validation_data["consumption"], alpha=0.95)
            interval_score_50_percent = self.interval_score_metric(future[f"pred_0.25"], future[f"pred_0.75"], validation_data["consumption"], alpha=0.5)

            result_validation[f"Interval Score 95% for iteration {i}"] = interval_score_95_percent
            result_validation[f"Interval Score 50% for iteration {i}"] = interval_score_50_percent


            print(f"Pinball loss for Validation from {validation_start} to {validation_end}: {np.mean(pinnball_loss_each_quantile)}")
            print(f"D2 Pinball score for Validation from {validation_start} to {validation_end}: {np.mean(d2_pinball_loss_each_quantile)}")

            pinnball_loss_each_iteration.append(np.mean(pinnball_loss_each_quantile))
            d2_pinball_loss_each_iteration.append(np.mean(d2_pinball_loss_each_quantile))
            interval_score_50_percent_each_iteration.append(interval_score_50_percent)
            interval_score_95_percent_each_iteration.append(interval_score_95_percent)
            
            validation_end = validation_start - pd.Timedelta(weeks = 1) # Set the new validation end to the start of the previous validation start

            # Sharpness: Compute interval width for middle intervals
            for lower, upper in [(0.025, 0.975), (0.25, 0.75)]:
                interval_width = future[f"pred_{upper}"] - future[f"pred_{lower}"]
                sharpness_for_iteration.append(interval_width.mean())
        
            sharpness_each_iteration.append(np.mean(sharpness_for_iteration))
            result_validation[f"Sharpness for iteration {i}"] = np.mean(sharpness_for_iteration)

            # Change the validation start into the future for the next iteration
            validation_start = validation_start + pd.Timedelta(weeks= 1)

        overall_pinball_loss = np.mean([result_validation[key] for key in result_validation.keys() if "Pinball" in key])
        overall_d2_pinball_score = np.mean([result_validation[key] for key in result_validation.keys() if "D2" in key])
        interval_score_95_percent = np.mean(interval_score_95_percent_each_iteration)
        interval_score_50_percent = np.mean(interval_score_50_percent_each_iteration)


        for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]:
            result_validation[f"Overall Pinball loss for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "Pinball loss for iteration" in key]
            )
            result_validation[f"Overall D2 Pinball score for quantile {quantile}"] = np.mean(
                [result_validation[key] for key in result_validation.keys() if f"quantile {quantile}" in key and "D2 Pinball score for iteration" in key]
            )

        overall_calibration = {quantile: np.mean(calibration_each_iteration[quantile]) for quantile in [0.025, 0.25, 0.5, 0.75, 0.975]}
        overall_sharpness = np.mean(sharpness_each_iteration)

        result_validation["Overall Pinball loss"] = overall_pinball_loss
        result_validation["Overall D2 Pinball score"] = overall_d2_pinball_score
        result_validation["Interval Score 95%"] = interval_score_95_percent
        result_validation["Interval Score 50%"] = interval_score_50_percent
        result_validation["Overall Calibration"] = overall_calibration
        result_validation["Overall Sharpness"] = overall_sharpness

        print(f"Overall Pinball loss: {overall_pinball_loss}")
        print(f"Overall D2 Pinball score: {overall_d2_pinball_score}")
        print(f"Interval Score 95%: {interval_score_95_percent}")
        print(f"Interval Score 50%: {interval_score_50_percent}")
        print(f"Overall Calibration: {overall_calibration}")
        print(f"Overall Sharpness: {overall_sharpness}")


        return result_validation, future_dict, data_dict



