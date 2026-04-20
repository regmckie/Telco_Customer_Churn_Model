import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Performs basic cleaning of the Telco dataset.
        - Trim column names
        - Drop ID column
        - Make 'TotalCharges' column numeric
        - Map target column 'Churn' to 0/1
        - Simple NA handling
    :param df: DataFrame of Telco dataset
    :param target_column: "Churn" column (i.e., the column we are ultimately predicting)
    :return: preprocessed DataFrame of Telco dataset
    """
    # Tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # Drop IDs if present (they're not really important when predicting churn)
    for column in ["customerID", "CustomerID", "customer_id"]:
        if column in df.columns:
            df = df.drop(columns=[column])

    # Makes Yes/No features into 0/1 (i.e., make them numeric)
    if target_column in df.columns and df[target_column].dtype == "object":
        df[target_column] = df[target_column].str.strip().map({"No": 0, "Yes": 1})

    # 'TotalCharges' often has blanks in the dataset --> coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # errors="corerce" says that if there's a value that can't be directly converted into a number (e.g., empty spaces, " ", etc.), then replace that value with NaN
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 'SeniorCitizen' should be 0/1 ints if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)  # Defaulting to 0 because that value appeared more than 1

    # Our simple NA strategy:
    # - Numeric values: fill with 0
    # - Others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df