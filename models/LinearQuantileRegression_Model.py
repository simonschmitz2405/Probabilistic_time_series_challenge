from processing.DataPreparing import DataPreparing

import pandas as pd
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import d2_pinball_score
from sklearn.linear_model import QuantileRegressor
from sklearn.impute import SimpleImputer

class LinearQuantileRegression_Model:
    """Class to train a linear quantile regression model for each quantile.

    Attributes:
        None

    Functions:
        quantile_regression_model: Train a Quantile Regressor for each quantile and return the models.
        quantile_regression_model_iterative: Train a Quantile Regressor for each quantile and return the models.
    """
    def quantile_regression_model(self, data, FEATURES, TARGET, fit_intercept, alpha, quantiles, n_splits, test_size):
        """Train a Quantile Regressor for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.
            FEATURES (list): List of feature column names.
            TARGET (str): Name of the target column.
            quantiles (list): List of quantiles for quantile regression.

        Returns:
            models (dict): A dictionary containing the trained models.
            results (dict): A dictionary containing evaluation metrics.
        """
        # Convert the fit_intercept parameter to a boolean because of optuna hyperparameter optimization
        print(f"-----Train the linear quantile regression model for target {TARGET}----")
        if fit_intercept == 0:
            fit_intercept = False
        elif fit_intercept == 1:
            fit_intercept = True

        data_preparer = DataPreparing()

        # Creates the Time Series cross validation
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        predictions = {} # Store the predicitons for ensemble model
        results = {}

        # Store the scores for each quantile
        pinball_loss_each_quantile = []
        relative_pinnball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            data_copy = data.copy()

            # Store the scores for each split
            pinball_loss_each_split = []
            relative_pinnball_loss_each_split = []
            d2_pinball_score_each_split = []

            split_counter = 0
            for train_index, test_index in tss.split(data_copy):

                data_copy = data_preparer.create_features_iterative(data_copy, TARGET)

                train = data_copy.iloc[train_index]
                test = data_copy.iloc[test_index]

                # Use mean imputation for missing values
                imputer = SimpleImputer(strategy='mean')  
                train_transformed= imputer.fit_transform(train[FEATURES[quantile]])
                test_transformed = imputer.transform(test[FEATURES[quantile]])

                train = train.copy()
                test = test.copy()

                train[FEATURES[quantile]] = pd.DataFrame(
                    train_transformed,
                    columns=FEATURES[quantile],
                    index=train.index
                )

                test[FEATURES[quantile]] = pd.DataFrame(
                    test_transformed,
                    columns=FEATURES[quantile],
                    index=test.index
                )

                # Feature and target extraction
                X_train = train[FEATURES[quantile]]
                y_train = train[TARGET]

                X_test = test[FEATURES[quantile]]
                y_test = test[TARGET]

                # Initialize and train the Quantile Regressor (train a model for each quantile)
                model = QuantileRegressor(quantile=quantile, solver='highs', alpha=alpha[quantile], fit_intercept=fit_intercept) 
                model.fit(X_train, y_train)
                models[f"model_{quantile}"] = model

                y_pred = model.predict(X_test)
                predictions[f"pred_{quantile}_{split_counter}"] = y_pred

                # Compute the pinball loss for each time series split
                pinball_loss = mean_pinball_loss(y_test, y_pred, alpha=quantile)
                pinball_loss_each_split.append(pinball_loss)

                # Compute the relative pinball loss for each time series split
                mean_y_test = np.mean(y_test)
                relative_pinball_loss = pinball_loss / mean_y_test
                relative_pinnball_loss_each_split.append(relative_pinball_loss)

                # Compute the d2 pinball score for each time series split
                d2_pinball_score_value = d2_pinball_score(y_test, y_pred, alpha=quantile)
                d2_pinball_score_each_split.append(d2_pinball_score_value)

                # Increase the split counter
                split_counter += 1

            # Compute the mean pinball loss for each quantile
            pinball_loss_each_quantile.append(np.mean(pinball_loss_each_split))
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(pinball_loss_each_split)

            # Compute the mean relative pinball loss for each quantile
            relative_pinnball_loss_each_quantile.append(np.mean(relative_pinnball_loss_each_split))
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(relative_pinnball_loss_each_split)

            # Compute the mean d2 pinball score for each quantile
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(d2_pinball_score_each_split)

            

        # Compute the overall mean pinball loss and d2 pinball score
        print(f"Mean pinball loss: {np.mean(pinball_loss_each_quantile)}")
        print(f"Mean relative pinball loss: {np.mean(relative_pinnball_loss_each_quantile)}")
        print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")

        overall_pinball_loss = np.mean(pinball_loss_each_quantile)
        overall_relative_pinball_loss = np.mean(relative_pinnball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(results)

        return models, predictions, results
    

    def quantile_regression_model_final(self, data, FEATURES, TARGET, fit_intercept, alpha, quantiles):
        """Train a Quantile Regressor for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.
            FEATURES (list): List of feature column names.
            TARGET (str): Name of the target column.
            quantiles (list): List of quantiles for quantile regression.

        Returns:
            models (dict): A dictionary containing the trained models.
            results (dict): A dictionary containing evaluation metrics.
        """
        # Convert the fit_intercept parameter to a boolean because of optuna hyperparameter optimization
        print(f"-----Train the linear quantile regression model for target {TARGET}----")
        if fit_intercept == 0:
            fit_intercept = False
        elif fit_intercept == 1:
            fit_intercept = True

        data_preparer = DataPreparing()

        models = {}


        for quantile in quantiles:
            data_copy = data.copy()



            data_copy = data_preparer.create_features_iterative(data_copy, TARGET)


            # Use mean imputation for missing values
            imputer = SimpleImputer(strategy='mean')  
            train_transformed= imputer.fit_transform(data_copy[FEATURES[quantile]])


            data_copy[FEATURES[quantile]] = pd.DataFrame(
                train_transformed,
                columns=FEATURES[quantile],
                index=data_copy.index
            )


            # Feature and target extraction
            X_train = data_copy[FEATURES[quantile]]
            y_train = data_copy[TARGET]


            # Initialize and train the Quantile Regressor (train a model for each quantile)
            model = QuantileRegressor(quantile=quantile, solver='highs', alpha=alpha[quantile], fit_intercept=fit_intercept) 
            model.fit(X_train, y_train)
            models[f"model_{quantile}"] = model


        return models

    def quantile_regression_model_iterative(self, data, FEATURES, TARGET, fit_intercept, alpha, quantiles):
            """Train a Quantile Regressor for each quantile and return the models.

            Args:
                data (pd.DataFrame): The data to train the models on.
                FEATURES (list): List of feature column names.
                TARGET (str): Name of the target column.
                quantiles (list): List of quantiles for quantile regression.

            Returns:
                models (dict): A dictionary containing the trained models.
                results (dict): A dictionary containing evaluation metrics.
            """
            # Convert the fit_intercept parameter to a boolean because of optuna hyperparameter optimization
            print("-----Train the linear quantile regression model-----")
            if fit_intercept == 0:
                fit_intercept = False
            elif fit_intercept == 1:
                fit_intercept = True

            data_preparer = DataPreparing()

            tss = TimeSeriesSplit(n_splits=5, test_size=30, gap=0)

            models = {}
            predictions = {} # Store the predicitons for ensemble model
            results = {}

            # Store the scores for each quantile
            pinball_loss_each_quantile = []
            relative_pinnball_loss_each_quantile = []
            d2_pinball_score_each_quantile = []

            for quantile in quantiles:
                # Drop rows with NaN values in the target column
                data_copy = data.copy()

                # Store the scores for each split
                pinball_loss_each_split = []
                relative_pinnball_loss_each_split = []
                d2_pinball_score_each_split = []

                split_counter = 0
                for train_index, test_index in tss.split(data_copy):

                    train = data_copy.iloc[train_index]
                    test = data_copy.iloc[test_index]

                    train = data_preparer.create_features_iterative(train, TARGET)

                    # Use mean imputation for missing values
                    imputer = SimpleImputer(strategy='mean')  
                    train_transformed= imputer.fit_transform(train[FEATURES[quantile]])
                    train[FEATURES[quantile]] = pd.DataFrame(
                        train_transformed,
                        columns=FEATURES[quantile],
                        index=train.index
                    )

                    # Feature and target extraction
                    X_train = train[FEATURES[quantile]]
                    y_train = train[TARGET]

                    # Initialize and train the Quantile Regressor (train a model for each quantile)
                    model = QuantileRegressor(quantile=quantile, solver='highs', alpha=alpha[quantile], fit_intercept=fit_intercept)
                    model.fit(X_train, y_train)
                    models[f"model_{quantile}"] = model

                    
                    y_pred_list = []
                    y_true_list = []
                    for i in range(len(test)):
                        # Take the current test data
                        test_current = test.iloc[:i+1].copy() 

                        # Merge the current test data with the train data to create the features
                        test_current = pd.concat([train, test_current])
                        test_current = data_preparer.create_features_iterative(test_current, TARGET)

                        # Take the respective rows which are the current test data
                        test_current = test_current.iloc[-(i+1):]
                        # print(f"Iteration: {i}")
                        # print(test_current)

                        # If there is a NaN in features columns fill with mean value
                        test_current_transformed = imputer.fit_transform(test_current[FEATURES[quantile]])
                        test_current[FEATURES[quantile]] = pd.DataFrame(
                            test_current_transformed,
                            columns=FEATURES[quantile],
                            index=test_current.index
                        )

                        X_test_current = test_current[FEATURES[quantile]]
                        y_test_current = test_current[TARGET]

                        y_pred_current = model.predict(X_test_current)

                        # Store the last prediction and the true value which is the current day
                        y_pred_list.append(y_pred_current[-1])
                        y_true_list.append(y_test_current.to_numpy()[-1])
                    
                    predictions[f"pred_{quantile}_{split_counter}"] = y_pred_list

                    pinnball_loss = mean_pinball_loss(y_true_list, y_pred_list, alpha=quantile)
                    relative_pinball_loss = pinnball_loss / np.mean(y_true_list)
                    d2_pinball_score_value = d2_pinball_score(y_true_list, y_pred_list, alpha=quantile)

                    pinball_loss_each_split.append(pinnball_loss)
                    relative_pinnball_loss_each_split.append(relative_pinball_loss)
                    d2_pinball_score_each_split.append(d2_pinball_score_value)

                    split_counter += 1

                # Compute the mean pinball loss for each quantile
                pinball_loss_each_quantile.append(np.mean(pinball_loss_each_split))
                results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(pinball_loss_each_split)

                # Compute the mean relative pinball loss for each quantile
                relative_pinnball_loss_each_quantile.append(np.mean(relative_pinnball_loss_each_split))
                results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(relative_pinnball_loss_each_split)

                # Compute the mean d2 pinball score for each quantile
                d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))
                results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(d2_pinball_score_each_split)

                

            # Compute the overall mean pinball loss and d2 pinball score
            print(f"Mean pinball loss: {np.mean(pinball_loss_each_quantile)}")
            print(f"Mean relative pinball loss: {np.mean(relative_pinnball_loss_each_quantile)}")
            print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")

            overall_pinball_loss = np.mean(pinball_loss_each_quantile)
            overall_relative_pinball_loss = np.mean(relative_pinnball_loss_each_quantile)
            overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

            results["Overall Pinball Loss"] = overall_pinball_loss
            results["Overall Relative Pinball Loss"] = overall_relative_pinball_loss
            results["Overall D2 Pinball Score"] = overall_d2_pinball_score

            print(results)

            return models, predictions, results