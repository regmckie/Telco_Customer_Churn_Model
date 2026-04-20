import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame

    :param file_path: Path to a CSV file
    :return: a pandas DataFrame; loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)