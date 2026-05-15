import pandas as pd
import numpy as np

def clean_data(df):
    """
    Perform data cleaning:
    - Remove missing descriptions
    - Remove cancelled orders
    - Remove non-positive quantities and prices
    - Convert InvoiceDate to datetime
    - Calculate TotalPrice
    """
    df = df.copy()
    
    # Drop missing descriptions
    df = df.dropna(subset=['Description'])
    
    # Identify cancelled orders (InvoiceNo starts with 'C')
    df['IsCancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
    
    # Filter for valid sales
    df = df[~df['IsCancelled']]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    # Convert dates and calculate price
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df

def create_features(df, frequency='D'):
    """
    Aggregate data and create time-series features.
    frequency: 'D' for Daily, 'W' for Weekly
    """
    # Aggregate to frequency
    if frequency == 'D':
        ts_data = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
    else:
        # For weekly, we set index to InvoiceDate then resample
        df_temp = df.copy()
        df_temp.set_index('InvoiceDate', inplace=True)
        ts_data = df_temp['TotalPrice'].resample(frequency).sum().reset_index()
        
    ts_data.columns = ['Date', 'Sales']
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])
    ts_data = ts_data.sort_values('Date')
    
    # Time-based features
    ts_data['Day'] = ts_data['Date'].dt.day
    ts_data['DayOfWeek'] = ts_data['Date'].dt.dayofweek
    ts_data['Month'] = ts_data['Date'].dt.month
    ts_data['Year'] = ts_data['Date'].dt.year
    ts_data['IsWeekend'] = ts_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lag features
    lags = [1, 2, 3, 7] if frequency == 'D' else [1, 2, 4]
    for lag in lags:
        ts_data[f'Sales_Lag_{lag}'] = ts_data['Sales'].shift(lag)
        
    # Rolling averages
    window = 7 if frequency == 'D' else 4
    ts_data[f'Sales_Rolling_{window}'] = ts_data['Sales'].shift(1).rolling(window=window).mean()
    
    return ts_data.dropna()
