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
    def mstl_model(
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
        print("-----Train the MSTL model-----")
        # Restrict the data to train the model on to the last timesteps_to_train
        data = data.iloc[-timesteps_to_train:]

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
        tss = TimeSeriesSplit(n_splits=5, test_size=7, gap=0)

        # quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
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
                    freq="d",
                )
                model.fit(df=train)

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

        return model, predicitons, results
