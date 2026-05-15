import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.preprocessing import clean_data, create_features
from src.visualization import plot_monthly_trend, plot_forecast_vs_actual, plot_future_forecast
from src.forecasting import train_model, evaluate_model, save_model, predict_recursive
from src.utils import ensure_directories, CLEAN_DATA_PATH
import pandas as pd

def run_pipeline():
    """Execute the full modular forecasting pipeline."""
    print("--- Initializing Sales Forecasting Pipeline ---")
    ensure_directories()
    
    # 1. Load
    print("Loading raw data...")
    raw_df = load_raw_data()
    
    # 2. Clean
    print("Cleaning data...")
    df = clean_data(raw_df)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    
    # 3. EDA
    print("Generating EDA visualizations...")
    plot_monthly_trend(df)
    
    # 4. Feature Engineering
    print("Engineering time-series features (Weekly)...")
    weekly_data = create_features(df, frequency='W')
    
    # 5. Training
    print("Training forecasting model...")
    test_size = 8
    train = weekly_data.iloc[:-test_size]
    test = weekly_data.iloc[-test_size:]
    
    X_train = train.drop(['Date', 'Sales'], axis=1)
    y_train = train['Sales']
    X_test = test.drop(['Date', 'Sales'], axis=1)
    y_test = test['Sales']
    
    model = train_model(X_train, y_train, model_type='RF')
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print(f"Model Training Complete. R2 Score: {metrics['R2']:.2f}")
    
    # 7. Forecast
    print("Generating 8-week future forecast...")
    plot_forecast_vs_actual(train, test, y_pred)
    
    future_dates = pd.date_range(start=weekly_data['Date'].iloc[-1] + pd.Timedelta(weeks=1), periods=8, freq='W')
    last_row_features = X_test.iloc[-1:].copy()
    future_forecast = predict_recursive(model, last_row_features, steps=8)
    
    plot_future_forecast(weekly_data, future_dates, future_forecast)
    save_model(model, 'weekly_sales_model.pkl')
    
    print("\nPipeline execution complete! Check 'outputs/' for results.")

if __name__ == "__main__":
    run_pipeline()
