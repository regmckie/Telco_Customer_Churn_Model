import os, sys
import pandas as pd

# Make 'src' folder importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/Raw-Telco-Customer-Churn-Dataset.csv"
OUT = "data/processed/Processed-Telco-Customer-Churn-Dataset.csv"

# STEP 1: LOAD RAW DATA
df = pd.read_csv(RAW)

# STEP 2: PREPROCESS DATA
df = preprocess_data(df, target_column="Churn")

# STEP 3: MAKE SURE 'Churn' IS NUMERIC 0/1 IF IT'S STILL AN OBJECT
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

# Perform sanity checks
assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

# STEP 4: FEATURE BUILDING
df_processed = build_features(df, target_column="Churn")

# STEP 5: SAVE
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")