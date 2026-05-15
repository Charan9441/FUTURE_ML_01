from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Online Retail.xlsx"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_data.csv"
DAILY_FEATURES_PATH = DATA_DIR / "daily_sales_features.csv"

# Model paths
MODEL_DIR = BASE_DIR / "models"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [DATA_DIR, MODEL_DIR, OUTPUT_DIR, CHART_DIR, PREDICTION_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    ensure_directories()
