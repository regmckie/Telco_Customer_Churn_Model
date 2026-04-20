import os
import pandas as pd

# Add the 'src' directory to the list of places Python searches for modules & packages
import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess_data import preprocess_data
from features.build_features import build_features

# --- CONFIG ---
DATA_PATH ="Users/R4G/Desktop/ML&DSProjects/Telco_Customer_Churn_Model/data/raw/Raw-Telco-Customer-Churn-Dataset.csv"
TARGET_COL = "Churn"


def main():
    print("*** TESTING PHASE 1: LOAD DATA --> PREPROCESS DATA --> BUILD FEATURES ***")

    # STEP 1: LOAD DATA
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(5))

    # STEP 2: PREPROCESS DATA
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_column=TARGET_COL)
    print(f"Data cleaned after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(5))

    # STEP 3: BUILD FEATURES
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_column=TARGET_COL)
    print(f"Data features built. Shape: {df_features.shape}")
    print(df_features.head(5))

    print("\n✅ PHASE 1 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()