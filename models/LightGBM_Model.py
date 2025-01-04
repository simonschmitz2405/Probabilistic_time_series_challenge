from data.DataPreparing import DataPreparing
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import d2_pinball_score
from typing import Tuple, Dict, List


class LightGBM_Model:
    def lightgbm_model(
        self,
        data: pd.DataFrame,
        FEATURES: dict[float, list[str]],
        TARGET: str,
        learning_rate: dict[float, float],
        num_estimators: dict[float, float],
        num_leaves: dict[float, float],
        reg_alpha: dict[float, float],
        quantiles: list[float],
    ) -> tuple[dict[str, lgb.LGBMRegressor], dict[str, float]]:
        """Train a LightGBM model for each quantile and return the models.

        Args:
            data (pd.DataFrame): The data to train the models on.

        Returns:
            model (dict): A dictionary containing the trained models.
        """
        print("-----Train the lightgbm model-----")

        data_preparer = DataPreparing()

        tss = TimeSeriesSplit(n_splits=5, test_size=7, gap=0)

        models = {}
        predictions = {}
        y_true_dict = {}
        results = {}

        # Store the metrices for each quantile
        pinnball_loss_each_quantile = []
        relative_pinnball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            # Store the scores for each split
            pinnball_loss_each_split = []
            relative_pinnball_loss_each_split = []
            d2_pinball_score_each_split = []

            split_counter = 0
            for train_index, test_index in tss.split(data):
                train = data.iloc[train_index]
                test = data.iloc[test_index]

                train = data_preparer.create_features(train)
                test = data_preparer.create_features(test)

                X_train = train[FEATURES[quantile]]
                y_train = train[TARGET]

                X_test = test[FEATURES[quantile]]
                y_test = test[TARGET]

                models[f"model_{quantile}"] = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    n_estimators=num_estimators[quantile],
                    learning_rate=learning_rate[quantile],
                    num_leaves=num_leaves[quantile],
                    reg_alpha=reg_alpha[quantile],
                    verbose=-1,
                )
                models[f"model_{quantile}"].fit(
                    X_train,
                    y_train,
                    eval_metric="quantile",
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    callbacks=[
                        # lgb.log_evaluation(period=50),
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                    ],
                )
                y_pred = models[f"model_{quantile}"].predict(X_test)
                predictions[f"pred_{quantile}_{split_counter}"] = y_pred

                y_true_dict[f"y_true_{split_counter}"] = np.array(y_test.tolist())

                # Compute the pinball loss for each time series split
                pinnball_loss = mean_pinball_loss(y_test, y_pred, alpha=quantile)
                pinnball_loss_each_split.append(pinnball_loss)

                # Compute the relative pinball loss for each time series split
                mean_y_test = np.mean(y_test)
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_split.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each time series split
                d2_pinnball_score = d2_pinball_score(y_test, y_pred, alpha=quantile)
                d2_pinball_score_each_split.append(d2_pinnball_score)

                # Increase the split counter
                split_counter += 1

            # Compute the mean pinball loss for each quantile
            pinnball_loss_each_quantile.append(np.mean(pinnball_loss_each_split))
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(
                pinnball_loss_each_split
            )

            # Compute the mean relative pinball loss for each quantile
            relative_pinnball_loss_each_quantile.append(
                np.mean(relative_pinnball_loss_each_split)
            )
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(
                relative_pinnball_loss_each_quantile
            )
            print(
                f"Quantile {quantile} Relative Pinball Loss: {np.mean(relative_pinnball_loss_each_split)}"
            )

            # Compute the mean d2 pinball score for each quantile
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(
                d2_pinball_score_each_split
            )

        # Compute the mean pinball loss
        print(f"Mean pinball loss: {np.mean(pinnball_loss_each_quantile)}")
        print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")
        print(
            f"Mean relative pinball loss: {np.mean(relative_pinnball_loss_each_quantile)}"
        )
        overall_pinnball_loss = np.mean(pinnball_loss_each_quantile)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(results)

        return models, predictions, results, y_true_dict

    def lightgbm_model_iterative(
        self,
        data: pd.DataFrame,
        FEATURES: Dict[float, List[str]],
        TARGET: str,
        learning_rate: Dict[float, float],
        num_estimators: Dict[float, float],
        num_leaves: Dict[float, float],
        reg_alpha: Dict[float, float],
        quantiles: List[float],
    ) -> Tuple[Dict[str, lgb.LGBMRegressor], Dict[str, float]]:
        """Train a LightGBM model for each quantile and return the models. This is iterative approach.

        Args:
            data (pd.DataFrame): The data to train the models on.
            FEATURES (dict): A dictionary mapping quantiles to lists of feature names.
            TARGET (str): The name of the target variable.
            learning_rate (dict): Learning rates for each quantile.
            num_estimators (dict): Number of estimators for each quantile.
            num_leaves (dict): Number of leaves for each quantile.
            reg_alpha (dict): L1 regularization for each quantile.
            quantiles (list): List of quantiles to predict.
            n_splits (int): Number of splits for TimeSeriesSplit.
            test_size (int): Size of the test set for each split.
            gap (int): Gap between train and test sets.

        Returns:
            tuple: A tuple containing models, predictions, results, and true values.
        """
        print("----- Train the LightGBM model -----")

        data_preparer = DataPreparing()

        tss = TimeSeriesSplit(n_splits=5, test_size=30, gap=0)

        models = {}
        predictions = {}
        y_true_dict = {}
        results = {}

        # Store the scores for each quantile
        pinnball_loss_each_quantile = []
        relative_pinnball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            # Store the scores for each split
            pinnball_loss_each_split = []
            relative_pinnball_loss_each_split = []
            d2_pinball_score_each_split = []

            split_counter = 0
            for train_index, test_index in tss.split(data):
                train = data.iloc[train_index]
                test = data.iloc[test_index]

                # Create features for the train set
                train = data_preparer.create_features_iterative(train)

                # Define the features and target for the train set
                X_train = train[FEATURES[quantile]]
                y_train = train[TARGET]

                # Define the model
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    n_estimators=num_estimators[quantile],
                    learning_rate=learning_rate[quantile],
                    num_leaves=num_leaves[quantile],
                    reg_alpha=reg_alpha[quantile],
                    verbose=-1,
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_metric="quantile",
                    eval_set=[
                        (X_train, y_train),
                        # (test[FEATURES[quantile]], test[TARGET]),
                    ],
                    # callbacks=[
                    #     lgb.early_stopping(stopping_rounds=50, verbose=False),
                    # ],
                )
                models[f"model_{quantile}_{split_counter}"] = model

                # Predict the next day iteratively
                y_pred_list = []
                y_true_list = []
                for i in range(len(test)):
                    test_current = test.iloc[
                        : i + 1
                    ].copy()  # Take rows from 0 to i (inclusive)
                    test_current = data_preparer.create_features_iterative(test_current)
                    X_test_current = test_current[FEATURES[quantile]]
                    y_test_current = test_current[TARGET]

                    y_pred_current = model.predict(X_test_current)
                    y_pred_list.append(y_pred_current[-1])
                    y_true_list.append(
                        y_test_current.iloc[-1]
                    )  # Use .iloc for single value

                predictions[f"pred_{quantile}_{split_counter}"] = y_pred_list
                y_true_dict[f"y_true_{split_counter}"] = np.array(y_true_list)

                # Compute pinball loss and other metrics
                pinnball_loss = mean_pinball_loss(
                    y_true_list, y_pred_list, alpha=quantile
                )
                relative_pinnball_loss = (
                    pinnball_loss / np.mean(y_true_list)

                )
                d2_pinnball_score_value = d2_pinball_score(
                    y_true_list, y_pred_list, alpha=quantile
                )

                pinnball_loss_each_split.append(pinnball_loss)
                relative_pinnball_loss_each_split.append(relative_pinnball_loss)
                d2_pinball_score_each_split.append(d2_pinnball_score_value)

                split_counter += 1

            # Store the evaluation metrics for each quantile
            pinnball_loss_each_quantile.append(np.mean(pinnball_loss_each_split))
            relative_pinnball_loss_each_quantile.append(
                np.mean(relative_pinnball_loss_each_split)
            )
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_each_split))

            # Store the overall results
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(
                pinnball_loss_each_split
            )
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(
                relative_pinnball_loss_each_split
            )
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(
                d2_pinball_score_each_split
            )

        # Compute the mean pinball loss
        overall_pinnball_loss = np.mean(pinnball_loss_each_quantile)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(f"Overall Pinnball Loss: {overall_pinnball_loss}")
        print(f"Overall Relative Pinnball Loss: {overall_relative_pinnball_loss}")
        print(f"Overall D2 Pinball Score: {overall_d2_pinball_score}")

        # Overall metrics
        print(f"Overall results: {results}")

        return models, predictions, results, y_true_dict
