from models.LightGBM_Model import LightGBM_Model
from models.MSTL_Model import MSTL_Model
from models.LinearQuantileRegression_Model import LinearQuantileRegression_Model
from models.GradientBoostingRegressor_Model import GradientBoostingRegressor_Model
from models.Baseline_Model import Baseline_Model

import optuna


class hyperparameter_optuna: # TODO FINISHI HERE THE CLASS 

    def optuna_objective(self, trial, data, feature_columns, selected_model, TARGET, quantile, n_splits, test_size):
        """Specific objective function for optuna hyperparameter search

        Args:
            trial (_type_): _description_
            data (_type_): _description_
            feature_columns (_type_): _description_
            selected_model (_type_): _description_
            TARGET (_type_): _description_

        Returns:
            result: Overall Pinball Loss
        """

        if selected_model == "lightgbm":
            # Hyperparameter 
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            num_leaves = trial.suggest_int("num_leaves", 2, 256)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 1.0)
            feature_mask = [trial.suggest_int(f"{feature}", 0, 1) for feature in feature_columns]

            selected_features = [feature for feature, mask in zip(feature_columns, feature_mask) if mask == 1]

            learning_rate = {quantile: learning_rate}
            n_estimators = {quantile: n_estimators}
            num_leaves = {quantile: num_leaves}
            reg_alpha = {quantile: reg_alpha}
            selected_features = {quantile: selected_features}

            # Get the Overall Pinball Loss
            lgbm = LightGBM_Model() 
            model, pred, result, y_true = lgbm.lightgbm_model(data, selected_features, TARGET, learning_rate, n_estimators, num_leaves, reg_alpha, [quantile], n_splits, test_size)

        if selected_model == "mstl":
            season_length_one = trial.suggest_int("season_length_one", 10, 100)

            season_length_one = {quantile: season_length_one}

            mstl = MSTL_Model() 
            model, pred, result = mstl.mstl_model_bike(data, 300, season_length_one, [quantile], n_splits, test_size)

        if selected_model == "linearquantileregression":
            lqr = LinearQuantileRegression_Model()
            # Hyperparameter
            alpha = trial.suggest_float("alpha", 0.00001, 0.1)
            feature_mask = [trial.suggest_int(f"{feature}", 0, 1) for feature in feature_columns]

            alpha = {quantile: alpha}

            selected_features = [feature for feature, mask in zip(feature_columns, feature_mask) if mask == 1]
            selected_features = {quantile: selected_features}

            model, pred, result = lqr.quantile_regression_model(data, selected_features, TARGET, 1, alpha, [quantile], n_splits, test_size)

        if selected_model == "gradientboostingregressor":
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            feature_mask = [trial.suggest_int(f"{feature}", 0, 1) for feature in feature_columns]

            learning_rate = {quantile: learning_rate}
            n_estimators = {quantile: n_estimators}

            selected_features = [feature for feature, mask in zip(feature_columns, feature_mask) if mask == 1]
            selected_features = {quantile: selected_features}

            gbr = GradientBoostingRegressor_Model()
            model, pred, result = gbr.gradientboostingregressor_model(data, selected_features, TARGET, learning_rate, n_estimators, [quantile], n_splits, test_size)

        # if selected_model == "baseline":
        #     baseline = Baseline_Model()
        #     pred, result = baseline.baseline_model(data, TARGET, [0.025, 0.25, 0.5, 0.75, 0.975], last_observations=100)

        return result["Overall Pinball Loss"]

    def tune_hyperparameters(self, data, feature_columns, selected_model, target , quantile, n_splits, test_size, n_trials=50):
        """
        Tune LightGBM hyperparameters using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.optuna_objective(trial, data, feature_columns, selected_model, target, quantile, n_splits, test_size),
            n_trials=n_trials,
        )
        print(f"Best hyperparameters: {study.best_params}")
        return study.best_params