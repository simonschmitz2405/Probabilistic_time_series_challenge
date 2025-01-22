from processing.DataPreparing import DataPreparing
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import d2_pinball_score

class Baseline_Model:
    """Baseline model to predict the future using the last 100 observations and the calculated quantiles.

    Attributes:
        None

    Functions:
        baseline_model: Baseline model that takes the last 100 observation and predicts using the calculated quantiles.
    """
    def baseline_model(self, data, TARGET, quantiles, last_observations=100):
        """Baseline model that takes the last 100 observation and predicts using the calculated quantiles.

        The baseline is also done using the Time Series Split for better comparison with the other models.
        """
        print("-----Train the baseline model-----")
        tss = TimeSeriesSplit(n_splits=5, test_size=30, gap=0) 

        # Quantiles to train models for
        predictions = {}
        results = {}

        # Lists to store overall metrics
        pinball_loss_each_quantile = []
        relative_pinball_loss_each_quantile = []
        d2_pinball_score_each_quantile = []

        for quantile in quantiles:
            pinball_loss_splits = []
            relative_pinnball_loss_splits = []
            d2_pinball_score_splits = []

            split_counter = 0
            for train_index, test_index in tss.split(data):
                train = data.iloc[train_index]
                test = data.iloc[test_index]

                # data_preparer = DataPreparing()
                # train = data_preparer.create_features(train, TARGET)
                # test = data_preparer.create_features(test, TARGET)

                y_pred = []
                y_true = []

                for i in range(len(test)):
                    context_data = train.iloc[-last_observations:] # Take the last 100 observations
                    quantile_prediction = context_data[TARGET].quantile(q = quantile)

                    # Store the true value and the prediction
                    y_pred.append(quantile_prediction)
                    y_true.append(test.iloc[i][TARGET])

                # Convert predicitons to numpy array
                y_pred = np.array(y_pred)
                y_true = np.array(y_true)

                predictions[f"pred_{quantile}_{split_counter}"] = y_pred

                # Evaluate
                pinball_loss = mean_pinball_loss(y_true, y_pred, alpha=quantile)
                relative_pinnball_loss = pinball_loss / np.mean(y_true)
                d2_pinball = d2_pinball_score(y_true, y_pred, alpha=quantile)

                # Append to lists
                pinball_loss_splits.append(pinball_loss)
                relative_pinnball_loss_splits.append(relative_pinnball_loss)
                d2_pinball_score_splits.append(d2_pinball)

                split_counter += 1

            # Compute the mean pinball loss for each quantile
            pinball_loss_each_quantile.append(np.mean(pinball_loss_splits))
            results[f"quantile_{quantile} Pinball Loss Mean"] = np.mean(pinball_loss_splits)

            # Compute the mean relative pinball loss for each quantile
            relative_pinball_loss_each_quantile.append(np.mean(relative_pinnball_loss_splits))
            results[f"quantile_{quantile} Relative Pinball Loss Mean"] = np.mean(relative_pinnball_loss_splits)

            # Compute the mean d2 pinball score for each quantile
            d2_pinball_score_each_quantile.append(np.mean(d2_pinball_score_splits))
            results[f"quantile_{quantile} D2 Pinball Score Mean"] = np.mean(d2_pinball_score_splits)

        # Compute the mean pinball loss
        print(f"Mean pinball loss: {np.mean(pinball_loss_each_quantile)}")
        print(f"Mean relative pinball loss: {np.mean(relative_pinball_loss_each_quantile)}")
        print(f"Mean d2 pinball score: {np.mean(d2_pinball_score_each_quantile)}")
        overall_pinnball_loss = np.mean(pinball_loss_each_quantile)
        overall_relative_pinnball_loss = np.mean(relative_pinball_loss_each_quantile)
        overall_d2_pinball_score = np.mean(d2_pinball_score_each_quantile)

        results["Overall Pinball Loss"] = overall_pinnball_loss
        results["Overall Relative Pinball Loss"] = overall_relative_pinnball_loss
        results["Overall D2 Pinball Score"] = overall_d2_pinball_score

        print(results)

        return predictions, results
    
