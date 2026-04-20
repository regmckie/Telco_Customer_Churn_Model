import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def tune_model(X, y):
    """
    Tunes an XGBoost model using Optuna.

    :param X: Features from dataset
    :param y: Target from dataset
    :return: Best hyperparameter values based on Optuna study
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")  # cv=3 is 3-fold cross validation; scoring="recall" measures recall for each fold (default is accuracy)
        return scores.mean()

    study = optuna.create_study(direction="maximize")  # 'maximize' because a higher recall is better
    study.optimize(objective, n_trials=20)  # Perform 20 trials; in each trial: pick hyperparams, train the model, calculate recall, keep track of best result

    print("Best Params: ", study.best_params)
    return study.best_params