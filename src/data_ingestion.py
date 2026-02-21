import pandas as pd

def load_data(file_path: str):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(file_path)
    return data
