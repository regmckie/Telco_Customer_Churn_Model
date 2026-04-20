"""
Runs the pipeline in this sequence: Load data --> Validate --> Preprocess --> Feature engineering
"""

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""

import os
import sys
import json
import time
import joblib
import mlflow
import argparse
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from jedi.inference import ValueSet
from posthog import project_root
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score

# --- FIX IMPORT PATH FOR LOCAL MODULES ---
# Allows imports from 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load local modules --> These are the pipeline components we wrote
from src.data.load_data import load_data  # Data loading w/ error handling
from src.data.preprocess_data import preprocess_data  # Basic data cleaning
from src.features.build_features import build_features  # Feature engineering
from src.utils.validate_data import validate_telco_data  # Data quality validation


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    :param args: List of arguments to run the pipeline/train the model (e.g., target, threshold, test_size, etc.)
    :return: N/A
    """

    # --- MLflow SETUP: ESSENTIAL FOR EXPERIMENT TRACKING ---
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = Path(__file__).resolve().parent.parent
    mlruns_path = args.mlflow_uri or (project_root / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if it doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # Log hyperparameters and configuration
        # These parameters are necessary for model reproducibility
        mlflow.log_param("model", "xgboost")  # Model type for comparison
        mlflow.log_param("threshold", args.threshold)  # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)  # Train/test split ratio

        # --- STEP 1: DATA LOADING ---
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # --- STEP 2: VALIDATION ---
        # Important for production ML; validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed! Logged to MLflow")

        # --- STEP 3: DATA PREPROCESSING ---
        print("🔧 Preprocessing data...")
        df = preprocess_data(df)  # Basic data cleaning (handle missing values, fix data types)

        # Saved processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "Processed-Telco-Customer-Churn-Dataset.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # --- STEP 4: FEATURE ENGINEERING ---
        print("🛠️ Building features...")

        target_col = args.target
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")

        # Apply feature engineering transformations
        df_enc = build_features(df, target_column=target_col)  # Binary encoding + one-hot encoding

        # Covert boolean columns to ints for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # Save feature metadata for serving consistency
        # Ensures serving pipeline uses exact same features in the exact same order
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target_col]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use idential transformations
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target_col  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} features for serving consistency")

        # --- STEP 5: TRAIN/TEST SPLIT ---
        print("📊 Performing train/test split on data...")

        X = df_enc.drop(columns=[target_col])  # Feature matrix
        y = df_enc[target_col]  # Target vector

        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # Handling class imbalance
        # Calculate scale_pos_weight to handle imbalanced dataset
        # Tells XGBoost to give more weight to the minority class (in this case, churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # --- STEP 6: MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS ---
        print("🤖 Training XGBoost model...")

        # These hyperparameters were optimized through hyperparameter tuning
        # In prod, consider using hyperparameter optimization tools like Optuna
        model = XGBClassifier(
            n_estimators=301,  # Number of trees (optimized)
            learning_rate=0.034,  # Step size shrinkage (optimized)
            max_depth=7,  # Maximum tree depth (optimized)
            subsample=0.95,  # Sample ratio of training instances
            colsample_bytree=0.98,  # Sample ratio of features for each tree
            n_jobs=-1,  # Use all CPU cores
            random_state=42,  # Reproducible results
            eval_metric="logloss",  # Evaluation metric
            scale_pos_weight=scale_pos_weight  # Weight for positive class (in this case, churners)
        )

        # Train model and track training time
        start_train_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_train_time
        mlflow.log_metric("train_time", training_time)
        print(f"✅ Model trained in {training_time} seconds")

        # --- STEP 7: MODEL EVALUATION ---
        print("📊 Evaluating model performance...")

        start_pred_time = time.time()
        probabilities = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)

        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold --> more sensitive to churn (i.e., higher recall, lower precision)
        y_preds = (probabilities >= args.threshold).astype(int)
        pred_time = time.time() - start_pred_time
        mlflow.log_metric("pred_time", pred_time)

        # Log evaluation metrics to MLflow
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_preds)  # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_preds)  # Of actual churners, how many did we actually catch?
        f1 = f1_score(y_test, y_preds)  # Harmonic mean of precision & recall
        roc_auc = roc_auc_score(y_test, y_preds)  # Area under ROC curve (threshold-independent)

        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"🎯 Model Performance:")
        print(f"    Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"    F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # --- STEP 8: MODEL SERIALIZATION AND LOGGING ---
        print("💾 Saving model to MLflow...")
        # Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",  # Creates a 'model/' folder in MLflow run artifacts
        )
        print ("✅ Model saved to MLflow for servign pipeline!")

        # --- FINAL PERFORMANCE SUMMARY ---
        print(f"\n⏱️  Performance Summary:")
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Inference time: {pred_time:.4f}s")
        print(f"    Samples per second: {len(X_test) / pred_time:.0f}")

        print(f"\n📈 Classification Report:")
        print(classification_report(y_test, y_preds, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)