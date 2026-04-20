import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model(df: pd.DataFrame, target_column: str):
    """
    Trains an XGBoost model and logs it with MLflow.

    :param df: The Telco Customer Churn dataset
    :param target_column: Name of the target column (Churn)
    :return: N/A
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)

        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.xgboost.log_model(model, "model")

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")