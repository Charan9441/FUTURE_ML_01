import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import CHART_DIR

# Set style
sns.set_theme(style="whitegrid")
plt.style.use('ggplot')

def plot_monthly_trend(df, save_name='monthly_sales_trend.png'):
    """Plot monthly total sales trend."""
    df_temp = df.copy()
    df_temp['Month'] = df_temp['InvoiceDate'].dt.to_period('M')
    monthly_sales = df_temp.groupby('Month')['TotalPrice'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line', marker='o', color='#2ecc71', linewidth=2)
    plt.title('Monthly Total Sales Trend', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    
    save_path = CHART_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def plot_forecast_vs_actual(train, test, y_pred, title='Sales Forecast vs Actual', save_name='forecast_validation.png'):
    """Plot training data, actual test data, and predicted values."""
    plt.figure(figsize=(14, 7))
    plt.plot(train['Date'], train['Sales'], label='Historical (Train)', color='#3498db', linewidth=2)
    plt.plot(test['Date'], test['Sales'], label='Actual (Test)', color='#2ecc71', linewidth=2, marker='o')
    plt.plot(test['Date'], y_pred, label='Forecasted', color='#e74c3c', linestyle='--', linewidth=2, marker='x')
    
    plt.title(title, fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Sales ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = CHART_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def plot_future_forecast(historical, future_dates, future_forecast, save_name='future_forecast.png'):
    """Plot historical data along with future projections."""
    plt.figure(figsize=(14, 7))
    plt.plot(historical['Date'], historical['Sales'], label='Historical', color='#3498db')
    plt.plot(future_dates, future_forecast, label='Future Forecast', color='#f1c40f', linestyle='--', marker='s')
    
    plt.title('Future Sales Demand Forecast', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Projected Sales ($)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = CHART_DIR / save_name
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path
