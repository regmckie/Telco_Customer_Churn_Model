import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


print("*** TESTING PHASE 2: MODELING WITH XGBOOST ***")

df = pd.read_csv("data/processed/Processed-Telco-Customer-Churn-Dataset.csv")

# Churn column must be a numeric 0/1
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"  # Checks if Churn has missing values
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"  # Checks if the only numeric values in Churn are 0/1

# Prepare X and y data
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

THRESHOLD = 0.4


def objective(trial):
    # Use the same hyperparameters as we did in the ExploratoryDataAnalysis.ipynb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),  # Number of trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),  # Controls how fast the model learns
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # Max depth of each tree
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # Fraction of training data used per tree
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Fraction of features used per tree
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # Controls how easily the model splits nodes (higher --> more conservative splits)
        "gamma": trial.suggest_float("gamma", 0, 5),  # Minimum loss reduction required to make a split (higher --> fewer splits and simpler model)
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),  # Encourages sparsity (feature selection effect)
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),  # Penalizes large weights to help prevent overfitting
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),  # For class imbalance
        "eval_metric": "logloss",  # Log loss used as evaluation metric
    }

    model = XGBClassifier(**params)  # Create XGBoost classifier with the sampled parameters
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    y_preds = (probabilities >= THRESHOLD).astype(int)

    return recall_score(y_test, y_preds, pos_label=1)

#  Run Optuna
study = optuna.create_study(direction="maximize")  # 'maximize' because a higher recall is better
study.optimize(objective, n_trials=30)  # Perform 30 trials; in each trial: pick hyperparams, train the model, calculate recall, keep track of best result

print("\nBest Params:", study.best_params)
print("Best Recall:", study.best_value)