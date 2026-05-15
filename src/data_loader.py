import pandas as pd
from pathlib import Path
from src.utils import RAW_DATA_PATH, CLEAN_DATA_PATH

def load_raw_data(custom_path=None):
    """Load the raw Excel dataset. Supports custom paths for Colab."""
    path = Path(custom_path) if custom_path else RAW_DATA_PATH
    if not path.exists():
        # Fallback for Colab default location
        colab_path = Path("/content/Online Retail.xlsx")
        if colab_path.exists():
            path = colab_path
        else:
            raise FileNotFoundError(f"Raw data not found at {path} or {colab_path}")
    return pd.read_excel(path)

def load_custom_file(file_obj):
    """Load an uploaded file object (CSV or Excel) from Streamlit."""
    try:
        if file_obj.name.endswith('.csv'):
            return pd.read_csv(file_obj)
        else:
            return pd.read_excel(file_obj)
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def load_cleaned_data():
    """Load the cleaned CSV dataset."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {CLEAN_DATA_PATH}")
    return pd.read_csv(CLEAN_DATA_PATH)

if __name__ == "__main__":
    try:
        df = load_raw_data()
        print(f"Loaded raw data with shape: {df.shape}")
    except Exception as e:
        print(e)
