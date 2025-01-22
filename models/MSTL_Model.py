import pandas as pd
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import d2_pinball_score
from statsmodels.tsa.seasonal import MSTL
import os
from statsforecast import StatsForecast
from statsforecast.models import MSTL

class MSTL_Model:
    """Class to train the MSTL model and to generate the forecast.

    Attributes:
        None

    Functions:
        mstl_model: Train the MSTL model for each quantile and return the models.

    """
    def mstl_model_bike(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        season_parameter_two: dict[float, int],
        quantiles: list[float],
        n_splits: int,
        test_size: int,
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """Train the MSTL model for each quantile and return the models.

        Args:
            data (_type_): _description_
            timesteps_to_train (int): _description_
            season_parameter_one (dict): _description_
            season_parameter_two (dict): _description_
            quantiles (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        print("-----Train the MSTL model for bike count-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["bike_count"]
        data_mstl.drop(columns=["bike_count"], inplace=True)

        # Split the data_mstl into training and test set
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        results = {}
        predicitons = {}
        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        # Store the scores for each split
        pinnball_loss_each_split = []
        relative_pinnball_loss_each_split = []
        d2_pinball_score_each_split = []

        split_counter = 0
        for train_index, test_index in tss.split(data_mstl):

            train = data_mstl.iloc[train_index]
            test = data_mstl.iloc[test_index]

            pinball_loss_each_quantile = []
            relative_pinnball_loss_each_quantile = []
            d2_pinball_score_each_quantile = []

            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                                season_parameter_two[quantile],
                                # season_parameter_three[quantile],
                            ]
                        )
                    ],
                    freq="d",
                )
                model.fit(df=train)
                models[f"model_{quantile}"] = model

                y_pred = model.forecast(df=train, h=7, level=[50, 95])
                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred[
                    quantile_dict[quantile]
                ].tolist()

                # Compute the pinball loss for each quantile
                pinnball_loss = mean_pinball_loss(
                    test["y"].tolist(),
                    y_pred[quantile_dict[quantile]].tolist(),
                    alpha=quantile,
                )
                pinball_loss_each_quantile.append(pinnball_loss)

                # Compute the relative pinball loss for each time series quantile
                mean_y_test = np.mean(test["y"].tolist())
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_quantile.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each quantile
                d2_pinnball_score = d2_pinball_score(
                    test["y"].tolist(),
                    y_pred[quantile_dict[quantile]].tolist(),
                    alpha=quantile,
                )
                d2_pinball_score_each_quantile.append(d2_pinnball_score)

            pinnball_loss_each_split.append(np.mean(pinball_loss_each_quantile))
            relative_pinnball_loss_each_split.append(
                np.mean(relative_pinnball_loss_each_quantile)
            )
            d2_pinball_score_each_split.append(np.mean(d2_pinball_score_each_quantile))

            split_counter += 1

        # Compute the mean pinball loss
        overall_pinnball_loss = np.mean(pinnball_loss_each_split)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_split)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score
        print(results)

        return models, predicitons, results
    

    def mstl_model_bike_final(
            self,
            data: pd.DataFrame,
            timesteps_to_train: int,
            season_parameter_one: dict[float, int],
            season_parameter_two: dict[float, int],
            quantiles: list[float],
        ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
            """Train the MSTL model for each quantile and return the models.

            Args:
                data (_type_): _description_
                timesteps_to_train (int): _description_
                season_parameter_one (dict): _description_
                season_parameter_two (dict): _description_
                quantiles (_type_): _description_

            Raises:
                ValueError: _description_

            Returns:
                _type_: _description_
            """
            print("-----Train the MSTL model for bike count-----")
            # Restrict the data to train the model on to the last timesteps_to_train
            data = data.iloc[-timesteps_to_train[0.025]:]

            os.environ["NIXTLA_ID_AS_COL"] = "True"

            data_mstl = data.copy()

            # Data preprocessing specific for the MSTL model
            data_mstl["ds"] = data_mstl.index
            data_mstl.reset_index(drop=True, inplace=True)
            data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
            data_mstl["unique_id"] = 1
            data_mstl["y"] = data_mstl["bike_count"]
            data_mstl.drop(columns=["bike_count"], inplace=True)

            models = {}


            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                                season_parameter_two[quantile],
                                # season_parameter_three[quantile],
                            ]
                        )
                    ],
                    freq="d",
                )
                model.fit(df=data_mstl)
                models[f"model_{quantile}"] = model
                

            return models


    def mstl_model_bike_iterative(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        quantiles: list[float],
        n_splits: int,
        test_size: int,
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """Train the MSTL model for each quantile and return the models.

        Args:
            data (_type_): _description_
            timesteps_to_train (int): _description_
            season_parameter_one (dict): _description_
            season_parameter_two (dict): _description_
            quantiles (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        print("-----Train the MSTL model for bike count-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["bike_count"]
        data_mstl.drop(columns=["bike_count"], inplace=True)

        # Split the data_mstl into training and test set
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        results = {}
        predicitons = {}
        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        # Store the scores for each split
        pinnball_loss_each_split = []
        relative_pinnball_loss_each_split = []
        d2_pinball_score_each_split = []

        split_counter = 0
        for train_index, test_index in tss.split(data_mstl):

            train = data_mstl.iloc[train_index]
            test = data_mstl.iloc[test_index]

            pinball_loss_each_quantile = []
            relative_pinnball_loss_each_quantile = []
            d2_pinball_score_each_quantile = []

            y_pred_list_dict = {}
            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                            ]
                        )
                    ],
                    freq="d",
                )
                model.fit(df=train)
                models[f"model_{quantile}"] = model

                y_pred_list = []
                y_true_list = []

                y_pred_list_dict[quantile_dict[quantile]] = y_pred_list
                for i in range(len(test)):
                    test_current = test.iloc[:i+1].copy()
                    test_current = pd.concat([train, test_current])

                    y_pred_current = models[f"model_{quantile}"].forecast(df=test_current, h=1, level=[50, 95])

                    y_pred_list_dict[quantile_dict[quantile]].append(y_pred_current[quantile_dict[quantile]].iloc[-1])
                    y_true_list.append(test_current["y"].iloc[-1])

                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred_list_dict[
                    quantile_dict[quantile]
                ]

                # Compute the pinball loss for each quantile
                pinnball_loss = mean_pinball_loss(
                    y_true_list,
                    y_pred_list_dict[quantile_dict[quantile]],
                    alpha=quantile,
                )
                pinball_loss_each_quantile.append(pinnball_loss)

                # Compute the relative pinball loss for each time series quantile
                mean_y_test = np.mean(y_true_list)
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_quantile.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each quantile
                d2_pinnball_score = d2_pinball_score(
                    y_true_list,
                    y_pred_list_dict[quantile_dict[quantile]],
                    alpha=quantile,
                )
                d2_pinball_score_each_quantile.append(d2_pinnball_score)

            pinnball_loss_each_split.append(np.mean(pinball_loss_each_quantile))
            relative_pinnball_loss_each_split.append(
                np.mean(relative_pinnball_loss_each_quantile)
            )
            d2_pinball_score_each_split.append(np.mean(d2_pinball_score_each_quantile))

            split_counter += 1

        # Compute the mean pinball loss
        overall_pinnball_loss = np.mean(pinnball_loss_each_split)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_split)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score
        print(results)

        return models, predicitons, results
    


    def mstl_model_energy(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        season_parameter_two: dict[float, int],
        quantiles: list[float],
        n_splits: int,
        test_size: int,
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """Train the MSTL model for each quantile and return the models.

        Args:
            data (_type_): _description_
            timesteps_to_train (int): _description_
            season_parameter_one (dict): _description_
            season_parameter_two (dict): _description_
            quantiles (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        print("-----Train the MSTL model for consumption-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["consumption"]
        data_mstl.drop(columns=["consumption"], inplace=True)

        # Split the data_mstl into training and test set
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)


        models = {}
        results = {}
        predicitons = {}
        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        # Store the scores for each split
        pinnball_loss_each_split = []
        relative_pinnball_loss_each_split = []
        d2_pinball_score_each_split = []

        split_counter = 0
        for train_index, test_index in tss.split(data_mstl):

            train = data_mstl.iloc[train_index]
            test = data_mstl.iloc[test_index]

            pinball_loss_each_quantile = []
            relative_pinnball_loss_each_quantile = []
            d2_pinball_score_each_quantile = []

            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                                season_parameter_two[quantile],
                            ]
                        )
                    ],
                    freq="h",
                )
                model.fit(df=train)
                models[f"model_{quantile}"] = model

                y_pred = model.forecast(df=train, h=test_size, level=[50, 95])
                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred[
                    quantile_dict[quantile]
                ].tolist()

                # Compute the pinball loss for each quantile
                pinnball_loss = mean_pinball_loss(
                    test["y"].tolist(),
                    y_pred[quantile_dict[quantile]].tolist(),
                    alpha=quantile,
                )
                pinball_loss_each_quantile.append(pinnball_loss)

                # Compute the relative pinball loss for each time series quantile
                mean_y_test = np.mean(test["y"].tolist())
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_quantile.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each quantile
                d2_pinnball_score = d2_pinball_score(
                    test["y"].tolist(),
                    y_pred[quantile_dict[quantile]].tolist(),
                    alpha=quantile,
                )
                d2_pinball_score_each_quantile.append(d2_pinnball_score)

            pinnball_loss_each_split.append(np.mean(pinball_loss_each_quantile))
            relative_pinnball_loss_each_split.append(
                np.mean(relative_pinnball_loss_each_quantile)
            )
            d2_pinball_score_each_split.append(np.mean(d2_pinball_score_each_quantile))

            split_counter += 1

        # Compute the mean pinball loss
        overall_pinnball_loss = np.mean(pinnball_loss_each_split)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_split)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score
        print(results)

        return models, predicitons, results
    

    def mstl_model_energy_final(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        season_parameter_two: dict[float, int],
        quantiles: list[float],
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """Train the MSTL model for each quantile and return the models.

        Args:
            data (_type_): _description_
            timesteps_to_train (int): _description_
            season_parameter_one (dict): _description_
            season_parameter_two (dict): _description_
            quantiles (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        print("-----Train the MSTL model for consumption-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["consumption"]
        data_mstl.drop(columns=["consumption"], inplace=True)



        models = {}

        for quantile in quantiles:
            model = StatsForecast(
                models=[
                    MSTL(
                        season_length=[
                            season_parameter_one[quantile],
                            season_parameter_two[quantile],
                        ]
                    )
                ],
                freq="h",
            )
            model.fit(df=data_mstl)
            models[f"model_{quantile}"] = model



        return models
    
    def mstl_model_energy_iterative(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        season_parameter_two: dict[float, int],
        quantiles: list[float],
        n_splits: int,
        test_size: int,
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """Train the MSTL model for each quantile and return the models.

        Args:
            data (_type_): _description_
            timesteps_to_train (int): _description_
            season_parameter_one (dict): _description_
            season_parameter_two (dict): _description_
            quantiles (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        print("-----Train the MSTL model for consumption-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        data_mstl = data.copy()

        # Data preprocessing specific for the MSTL model
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["consumption"]
        data_mstl.drop(columns=["consumption"], inplace=True)

        # Split the data_mstl into training and test set
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        results = {}
        predicitons = {}
        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        # Store the scores for each split
        pinnball_loss_each_split = []
        relative_pinnball_loss_each_split = []
        d2_pinball_score_each_split = []

        split_counter = 0
        for train_index, test_index in tss.split(data_mstl):

            train = data_mstl.iloc[train_index]
            test = data_mstl.iloc[test_index]

            pinball_loss_each_quantile = []
            relative_pinnball_loss_each_quantile = []
            d2_pinball_score_each_quantile = []

            y_pred_list_dict = {}
            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                                season_parameter_two[quantile]
                            ]
                        )
                    ],
                    freq="d",
                )
                model.fit(df=train)
                models[f"model_{quantile}"] = model

                y_pred_list = []
                y_true_list = []

                y_pred_list_dict[quantile_dict[quantile]] = y_pred_list
                for i in range(len(test)):
                    test_current = test.iloc[:i+1].copy()
                    test_current = pd.concat([train, test_current])

                    y_pred_current = models[f"model_{quantile}"].forecast(df=test_current, h=1, level=[50, 95])

                    y_pred_list_dict[quantile_dict[quantile]].append(y_pred_current[quantile_dict[quantile]].iloc[-1])
                    y_true_list.append(test_current["y"].iloc[-1])

                predicitons[f"pred_{quantile}_{split_counter}"] = y_pred_list_dict[
                    quantile_dict[quantile]
                ]

                # Compute the pinball loss for each quantile
                pinnball_loss = mean_pinball_loss(
                    y_true_list,
                    y_pred_list_dict[quantile_dict[quantile]],
                    alpha=quantile,
                )
                pinball_loss_each_quantile.append(pinnball_loss)

                # Compute the relative pinball loss for each time series quantile
                mean_y_test = np.mean(y_true_list)
                relative_pinnball_loss = pinnball_loss / mean_y_test
                relative_pinnball_loss_each_quantile.append(relative_pinnball_loss)

                # Compute the d2 pinball score for each quantile
                d2_pinnball_score = d2_pinball_score(
                    y_true_list,
                    y_pred_list_dict[quantile_dict[quantile]],
                    alpha=quantile,
                )
                d2_pinball_score_each_quantile.append(d2_pinnball_score)
                print(y_pred_list_dict[quantile_dict[quantile]])

            pinnball_loss_each_split.append(np.mean(pinball_loss_each_quantile))
            relative_pinnball_loss_each_split.append(
                np.mean(relative_pinnball_loss_each_quantile)
            )
            d2_pinball_score_each_split.append(np.mean(d2_pinball_score_each_quantile))

            split_counter += 1

        # Compute the mean pinball loss
        overall_pinnball_loss = np.mean(pinnball_loss_each_split)
        overall_relative_pinnball_loss = np.mean(relative_pinnball_loss_each_split)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score
        print(results)

        return models, predicitons, results


    def mstl_model_energy_iterative_optimized(
        self,
        data: pd.DataFrame,
        timesteps_to_train: int,
        season_parameter_one: dict[float, int],
        season_parameter_two: dict[float, int],
        quantiles: list[float],
        n_splits: int,
        test_size: int,
    ) -> tuple[dict[str, StatsForecast], dict[str, float]]:
        """
        Train and predict with the MSTL model for each quantile iteratively.

        Args:
            data (pd.DataFrame): Dataset containing energy consumption data.
            timesteps_to_train (int): Number of timesteps to use for training.
            season_parameter_one (dict): Season lengths for each quantile (primary).
            season_parameter_two (dict): Season lengths for each quantile (secondary).
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            n_splits (int): Number of splits for time series cross-validation.
            test_size (int): Number of timesteps for the test set.

        Returns:
            tuple: A dictionary of trained models, predictions, and evaluation results.
        """
        print("-----Train the MSTL model for consumption (Optimized)-----")
        # Restrict the data to the last `timesteps_to_train`
        data = data.iloc[-timesteps_to_train[0.025]:]

        os.environ["NIXTLA_ID_AS_COL"] = "True"

        # Prepare data for MSTL
        data_mstl = data.copy()
        data_mstl["ds"] = data_mstl.index
        data_mstl.reset_index(drop=True, inplace=True)
        data_mstl["ds"] = pd.to_datetime(data_mstl["ds"])
        data_mstl["unique_id"] = 1
        data_mstl["y"] = data_mstl["consumption"]
        data_mstl.drop(columns=["consumption"], inplace=True)

        # Set up time series splits
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        models = {}
        predictions = {}
        quantile_dict = {
            0.025: "MSTL-lo-95",
            0.25: "MSTL-lo-50",
            0.5: "MSTL",
            0.75: "MSTL-hi-50",
            0.975: "MSTL-hi-95",
        }

        # Store results for evaluation
        pinball_loss_per_split = []
        relative_pinball_loss_per_split = []
        d2_pinball_score_per_split = []

        split_counter = 0
        for train_index, test_index in tss.split(data_mstl):
            train = data_mstl.iloc[train_index]
            test = data_mstl.iloc[test_index]

            pinball_loss_per_quantile = []
            relative_pinball_loss_per_quantile = []
            d2_pinball_score_per_quantile = []

            for quantile in quantiles:
                model = StatsForecast(
                    models=[
                        MSTL(
                            season_length=[
                                season_parameter_one[quantile],
                                season_parameter_two[quantile],
                            ]
                        )
                    ],
                    freq="h",
                )
                model.fit(df=train)
                models[f"model_{quantile}"] = model

            # Iterative predictions for all quantiles
            y_true = test["y"].tolist()
            y_pred_dict = {q: [] for q in quantiles}

            for i in range(test_size):
                # Prepare rolling train data
                rolling_train = pd.concat([train, test.iloc[:i]])

                for quantile in quantiles:
                    model = models[f"model_{quantile}"]
                    forecast = model.forecast(df=rolling_train, h=1, level=[50, 95])
                    y_pred_dict[quantile].append(
                        forecast[quantile_dict[quantile]].iloc[-1]
                    )

            # Store predictions and evaluate
            for quantile in quantiles:
                y_pred = y_pred_dict[quantile]
                predictions[f"pred_{quantile}_{split_counter}"] = y_pred

                # Compute metrics
                pinball_loss = mean_pinball_loss(y_true, y_pred, alpha=quantile)
                pinball_loss_per_quantile.append(pinball_loss)

                mean_y_test = np.mean(y_true)
                relative_pinball_loss = pinball_loss / mean_y_test
                relative_pinball_loss_per_quantile.append(relative_pinball_loss)

                d2_score = d2_pinball_score(y_true, y_pred, alpha=quantile)
                d2_pinball_score_per_quantile.append(d2_score)

            pinball_loss_per_split.append(np.mean(pinball_loss_per_quantile))
            relative_pinball_loss_per_split.append(
                np.mean(relative_pinball_loss_per_quantile)
            )
            d2_pinball_score_per_split.append(np.mean(d2_pinball_score_per_quantile))

            split_counter += 1

        # Compute overall metrics
        results = {
            "Overall Pinball Loss": np.mean(pinball_loss_per_split),
            "Overall Relative Pinball Loss": np.mean(relative_pinball_loss_per_split),
            "Overall D2 Pinball Score": np.mean(d2_pinball_score_per_split),
        }
        print(results)

        return models, predictions, results

