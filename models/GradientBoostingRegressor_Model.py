from processing.DataPreparing import DataPreparing

import pandas as pd
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import d2_pinball_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingRegressor_Model:
    """Class to train a Gradient Boosting model for each quantile and return the models.

    Attributes:
        None

    Functions:
        gradientboostingregressor_model: Train a Gradient Boosting model for each quantile and return the models.
        gradientboostingregressor_model_iterative: Train a Gradient Boosting model for each quantile and return the models.
    """
    def gradientboostingregressor_model(self, data, FEATURES, TARGET, learning_rate, n_estimators, quantiles, n_splits, test_size):
        """Train a Gradient Boosting model for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.

        Returns:
            model (dict): A dictionary containing the trained models.
        """	
        print("-----Train the Gradient boosting regressor model-----")

        # Function to get the features
        data_preparer = DataPreparing()

        # Creates the Time Series cross validation
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        predicitons = {}
        results = {}

        # Store the scores for each quantile
        pinnball_loss_each_quantile = []
        relative_pinnball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            data_copy = data.copy()

            # Store the scores for each split
            pinnball_loss_each_split = []
            relative_pinnball_loss_each_split = []
            d2_pinball_score_each_split = []

            split_counter = 0
            for train_index, test_index in tss.split(data_copy):

                data_copy = data_preparer.create_features_iterative(data_copy, TARGET)

                train = data_copy.iloc[train_index]
                test = data_copy.iloc[test_index]

                # Use mean imputation for missing values green
                imputer = SimpleImputer(strategy='mean') 
                train_transformed = imputer.fit_transform(train[FEATURES[quantile]])
                test_transformed = imputer.transform(test[FEATURES[quantile]])

                train.loc[:,FEATURES[quantile]] = pd.DataFrame(
                    train_transformed,
                    columns=FEATURES[quantile],
                    index=train.index
                )

                test.loc[:,FEATURES[quantile]] = pd.DataFrame(
                    test_transformed,
                    columns=FEATURES[quantile],
                    index=test.index
                )

                X_train = train[FEATURES[quantile]]
                y_train = train[TARGET]

                X_test = test[FEATURES[quantile]]
                y_test = test[TARGET]

                model = GradientBoostingRegressor(loss='quantile', alpha = quantile, learning_rate = learning_rate[quantile], n_estimators = n_estimators[quantile])
                model.fit(X_train,y_train)
                models[f"model_{quantile}"] = model

                y_pred = model.predict(X_test)
                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred

                # Compute the pinball loss for each time series split
                pinnball_loss = mean_pinball_loss(y_test, y_pred, alpha=quantile)
                pinnball_loss_each_split.append(pinnball_loss)

                mean_y_test = np.mean(y_test)
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_split.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each time series split
                d2_pinnball_score = d2_pinball_score(y_test, y_pred, alpha=quantile)
                d2_pinball_score_each_split.append(d2_pinnball_score)

                split_counter += 1

            # Compute the mean pinball loss for each quantile
            pinnball_loss_each_quantile.append(np.mean(pinnball_loss_each_split))
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(pinnball_loss_each_split)

            # Compute the mean relative pinball loss for each quantile
            relative_pinnball_loss_each_quantile.append(np.mean(relative_pinnball_loss_each_split))
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(relative_pinnball_loss_each_split)

            # Compute the mean d2 pinball score for each quantile
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(d2_pinball_score_each_split)

        # Compute the mean pinball loss
        print(f"Mean pinball loss: {np.mean(pinnball_loss_each_quantile)}")
        print(f"Mean relative pinball loss: {np.mean(relative_pinnball_loss_each_quantile)}")
        print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")


        overall_pinnball_loss = np.mean(pinnball_loss_each_quantile)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(results)


        return models, predicitons, results


    def gradientboostingregressor_model_final(self, data, FEATURES, TARGET, learning_rate, n_estimators, quantiles):
        """Train a Gradient Boosting model for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.

        Returns:
            model (dict): A dictionary containing the trained models.
        """	
        print("-----Train the Gradient boosting regressor model-----")

        # Function to get the features
        data_preparer = DataPreparing()

        models = {}


        for quantile in quantiles:
            data_copy = data.copy()



            data_copy = data_preparer.create_features_iterative(data_copy, TARGET)


            # Use mean imputation for missing values green
            imputer = SimpleImputer(strategy='mean') 
            train_transformed = imputer.fit_transform(data_copy[FEATURES[quantile]])

            data.loc[:,FEATURES[quantile]] = pd.DataFrame(
                train_transformed,
                columns=FEATURES[quantile],
                index=data.index
            )



            X_train = data[FEATURES[quantile]]
            y_train = data[TARGET]


            model = GradientBoostingRegressor(loss='quantile', alpha = quantile, learning_rate = learning_rate[quantile], n_estimators = n_estimators[quantile])
            model.fit(X_train,y_train)
            models[f"model_{quantile}"] = model



        return models
    

    def gradientboostingregressor_model_iterative(self, data, FEATURES, TARGET, quantiles):
        """Train a Gradient Boosting model for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.

        Returns:
            model (dict): A dictionary containing the trained models.
        """	
        print("-----Train the Gradient boosting regressor model-----")

        # Function to get the features
        data_preparer = DataPreparing()

        # Creates the Time Series cross validation
        tss = TimeSeriesSplit(n_splits=5, test_size=30, gap=0)

        models = {}
        predicitons = {}
        results = {}

        # Store the scores for each quantile
        pinnball_loss_each_quantile = []
        relative_pinnball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            data_copy = data.copy()

            # Store the scores for each split
            pinnball_loss_each_split = []
            relative_pinnball_loss_each_split = []
            d2_pinball_score_each_split = []

            split_counter = 0
            for train_index, test_index in tss.split(data_copy):

                train = data_copy.iloc[train_index]
                test = data_copy.iloc[test_index]

                train = data_preparer.create_features_iterative(train, TARGET)

                # Use mean imputation for missing values green
                imputer = SimpleImputer(strategy='mean') 
                train_transformed = imputer.fit_transform(train[FEATURES[quantile]])
                train[FEATURES[quantile]] = pd.DataFrame(
                    train_transformed,
                    columns=FEATURES[quantile],
                    index=train.index
                )

                # Feature and target extraction
                X_train = train[FEATURES[quantile]]
                y_train = train[TARGET]


                model = GradientBoostingRegressor(loss='quantile', alpha= quantile)
                model.fit(X_train,y_train)
                models[f"model_{quantile}"] = model

                y_pred_list = []
                y_true_list = []
                for i in range(len(test)):
                    # Take current test data
                    test_current = test.iloc[:i+1].copy()

                    # Create iterativ the features for the current test by also taking the training data
                    test_current = pd.concat([train, test_current])
                    test_current = data_preparer.create_features_iterative(test_current, TARGET)

                    # Take the respective rows which are the current test data
                    test_current = test_current.iloc[-(i+1):]

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


                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred_list

                # Compute the pinball loss for each time series split
                pinnball_loss = mean_pinball_loss(y_true_list, y_pred_list, alpha=quantile)
                mean_y_test = np.mean(y_true_list)
                relative_pinnball_loss = pinnball_loss / mean_y_test
                d2_pinnball_score = d2_pinball_score(y_true_list, y_pred_list, alpha=quantile)

                pinnball_loss_each_split.append(pinnball_loss)
                relative_pinnball_loss_each_split.append(relative_pinnball_loss)
                d2_pinball_score_each_split.append(d2_pinnball_score)

                split_counter += 1

            # Compute the mean pinball loss for each quantile
            pinnball_loss_each_quantile.append(np.mean(pinnball_loss_each_split))
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(pinnball_loss_each_split)

            # Compute the mean relative pinball loss for each quantile
            relative_pinnball_loss_each_quantile.append(np.mean(relative_pinnball_loss_each_split))
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(relative_pinnball_loss_each_split)

            # Compute the mean d2 pinball score for each quantile
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(d2_pinball_score_each_split)

        # Compute the mean pinball loss
        print(f"Mean pinball loss: {np.mean(pinnball_loss_each_quantile)}")
        print(f"Mean relative pinball loss: {np.mean(relative_pinnball_loss_each_quantile)}")
        print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")


        overall_pinnball_loss = np.mean(pinnball_loss_each_quantile)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(results)


        return models, predicitons, results