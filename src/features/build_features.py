import pandas as pd


def _map_binary_series(ser: pd.Series) -> pd.Series:
    """
    Applies deterministic binary encoding to 2-category features.

    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings are deterministic
    and must be consistent between training and serving.

    :param ser: Series that represents the categorical feature
    :return: New series of the categorical feature that contains the binary encoding
    """
    # Get unique values and remove NaN
    values = list(pd.Series(ser.dropna().unique()).astype(str))
    valset = set(values)

    # --- DETERMINISTIC BINARY MAPPINGS ---
    # These exact mappings are hardcoded in serving pipeline because we know the names of these features & how they work

    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return ser.map({"No": 0, "Yes": 1}).astype("Int64")

    # Gender mapping
    if valset == {"Male", "Female"}:
        return ser.map({"Female": 0, "Male": 1}).astype("Int64")

    # --- GENERIC BINARY MAPPINGS ---
    # For any other 2-category feature, use stable alphabetical ordering
    # This generic mapping is if we don't know the names of the features or how they work
    if len(values) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(values)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return ser.astype(str).map(mapping).astype("Int64")

    # --- NON-BINARY FEATURES ---
    # Return them unchanged (will be handled later by one-hot encoding)

    return ser


def build_features(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Applies complete feature engineering pipeline for training data.

    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    :param df: DataFrame of Telco dataset
    :param target_column: "Churn" column (i.e., the column we are ultimately predicting)
    :return: DataFrame with newly engineered, ML-ready features
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # --- STEP 1: IDENTIFY FEATURE TYPES ---
    # Find categorical columns (object dtype) excluding the target variable
    obj_columns = [col for col in df.select_dtypes(include=["object"]).columns if col != target_column]
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"   📊 Found {len(obj_columns)} categorical and {len(numeric_columns)} numeric columns")

    # --- STEP 2: SPLIT CATEGORICAL BY CARDINALITY ---
    # Binary features (those with exactly 2 unique values) get binary encoding
    # Multi-category features (those with more than 2 unique values) get one-hot encoding
    binary_columns = [col for col in obj_columns if df[col].dropna().nunique() == 2]
    multi_columns = [col for col in obj_columns if df[col].dropna().nunique() > 2]

    print(f"   🔢 Binary features: {len(binary_columns)} | Multi-category features: {len(multi_columns)}")

    if binary_columns:
        print(f"        Binary: {binary_columns}")
    if multi_columns:
        print(f"        Multi-category: {multi_columns}")

    # --- STEP 3: APPLY BINARY ENCODING ---
    # Convert 2-category features to 0/1 using deterministic mappings
    for col in binary_columns:
        original_dtype = df[col].dtype
        df[col] = _map_binary_series(df[col].astype(str))
        print(f"      ✅ {col}: {original_dtype} --> binary (0/1)")

    # --- STEP 4: CONVERT BOOLEAN COLUMNS ---
    # XGBoost requires int inputs, not booleans
    bool_columns = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_columns:
        df[bool_columns] = df[bool_columns].astype(int)
        print(f"   🔄 Converted {len(bool_columns)} boolean columns to int: {bool_columns}")

    # --- STEP 5: ONE-HOT ENCODING FOR MULTI-CATEGORY FEATURES ---
    # NOTE: 'drop_first=True' prevents multicollinearity
    if multi_columns:
        print(f"   🌟 Applying one-hot encoding to {len(multi_columns)} multi-category columns...")

        # Apply one-hot encoding with drop_first=True
        df = pd.get_dummies(df, columns=multi_columns, drop_first=True)

        original_shape = df.shape
        new_features = df.shape[1] - original_shape[1] + len(multi_columns)
        print(f"      ✅ Created {new_features} new features from {len(multi_columns)} categorical columns")

    # --- STEP 6: DATA TYPE CLEANING ---
    # Convert the nullable ints (Int64) to standard ints for XGBoost
    for col in binary_columns:
        if pd.api.types.is_integer_dtype(df[col]):
            # Fill any NaN values with 0 and convert to int
            df[col] = df[col].fillna(0).astype(int)

    print(f"✅ Feature engineer complete: {df.shape[1]} final features")
    return df