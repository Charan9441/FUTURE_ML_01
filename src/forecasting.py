import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.utils import MODEL_DIR

def train_model(X_train, y_train, model_type='GBM'):
    """
    Train a regression model.
    model_type: 'GBM' (Gradient Boosting) or 'RF' (Random Forest)
    """
    if model_type == 'GBM':
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred):
    """Calculate and return performance metrics."""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def save_model(model, filename):
    """Save model to the models/ directory."""
    save_path = MODEL_DIR / filename
    joblib.dump(model, save_path)
    return save_path

def predict_recursive(model, last_row_features, steps=8):
    """
    Perform multi-step recursive forecasting.
    Note: This is a simplified version that assumes lag_1 is the primary recursive feature.
    """
    forecasts = []
    current_features = last_row_features.copy()
    
    for _ in range(steps):
        pred = model.predict(current_features)[0]
        forecasts.append(pred)
        
        # Update lag_1 for next step if it exists in features
        if 'Sales_Lag_1' in current_features.columns:
            current_features['Sales_Lag_1'] = pred
            
    return forecasts
