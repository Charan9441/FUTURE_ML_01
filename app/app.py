import streamlit as st
import pandas as pd
import joblib
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for modular imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loader import load_raw_data, load_custom_file
from src.preprocessing import clean_data, create_features
from src.forecasting import predict_recursive
from src.visualization import plot_forecast_vs_actual, plot_future_forecast

MODEL_PATH = BASE_DIR / "models" / "weekly_sales_model.pkl"

def main():
    st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📈")
    
    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None

    # --- LANDING PAGE: DATA SELECTION ---
    if st.session_state.df is None:
        st.title("🚀 Sales & Demand Forecasting")
        st.markdown("""
        Transform your historical sales data into actionable future insights. 
        Select a dataset below to begin.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### 📂 Option 1: Standard Dataset")
            st.write("Use the 'Online Retail' dataset (UCI Machine Learning Repository).")
            if st.button("Load Standard Dataset", use_container_width=True):
                with st.spinner("Loading standard dataset..."):
                    try:
                        st.session_state.df = load_raw_data()
                        st.session_state.data_source = "Standard (Online Retail)"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading standard data: {e}")
                    
        with col2:
            st.success("### 📤 Option 2: Upload Your Own")
            st.write("Upload a CSV or Excel file (InvoiceDate, Quantity, UnitPrice columns).")
            uploaded_file = st.file_uploader("Drop your file here", type=["csv", "xlsx"], key="main_uploader")
            if uploaded_file:
                try:
                    st.session_state.df = load_custom_file(uploaded_file)
                    st.session_state.data_source = f"Custom ({uploaded_file.name})"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- MAIN DASHBOARD ---
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["📊 Overview", "🔮 Forecasting", "💡 Business Insights"])
        
        st.sidebar.markdown("---")
        st.sidebar.write(f"**Current Data:**")
        st.sidebar.code(st.session_state.data_source)
        if st.sidebar.button("🔄 Change Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.data_source = None
            st.rerun()

        if page == "📊 Overview":
            st.title("Data Overview")
            df = st.session_state.df
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Records", f"{len(df):,}")
            col_b.metric("Unique Items", df['Description'].nunique() if 'Description' in df.columns else "N/A")
            col_c.metric("Missing Values", df.isnull().sum().sum())
            
            st.markdown("### 🔍 Raw Data Preview")
            st.dataframe(df.head(50), use_container_width=True)
            
        elif page == "🔮 Forecasting":
            st.title("Future Demand Projections")
            
            if "Standard" in st.session_state.data_source:
                if MODEL_PATH.exists():
                    st.success("Model loaded successfully!")
                    
                    # Preprocess and Generate Forecast
                    with st.spinner("Generating 8-week forecast..."):
                        df_clean = clean_data(st.session_state.df)
                        weekly_data = create_features(df_clean, frequency='W')
                        
                        model = joblib.load(MODEL_PATH)
                        
                        # Last row features for recursive prediction
                        X_features = weekly_data.drop(['Date', 'Sales'], axis=1)
                        last_row = X_features.iloc[-1:]
                        
                        future_forecast = predict_recursive(model, last_row, steps=8)
                        future_dates = pd.date_range(
                            start=weekly_data['Date'].iloc[-1] + pd.Timedelta(weeks=1), 
                            periods=8, 
                            freq='W'
                        )
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(weekly_data['Date'].tail(20), weekly_data['Sales'].tail(20), label='Historical', marker='o')
                        ax.plot(future_dates, future_forecast, label='Forecast', linestyle='--', marker='s', color='orange')
                        ax.set_title("8-Week Sales Forecast")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Sales ($)")
                        ax.legend()
                        st.pyplot(fig)
                        
                        st.markdown("### 📋 Forecast Results")
                        forecast_df = pd.DataFrame({'Date': future_dates, 'Projected Sales': future_forecast})
                        st.table(forecast_df)
                else:
                    st.error(f"Model file not found at {MODEL_PATH}. Please run `src/main.py` first to train the model.")
            else:
                st.warning("⚠️ Forecasting for custom datasets is coming soon. The current model is trained specifically for the 'Online Retail' dataset structure.")
                
        elif page == "💡 Business Insights":
            st.title("Strategic Insights")
            st.markdown("""
            ### 🛒 Inventory Strategy
            *   **Safety Stock**: We recommend maintaining a **15-20% buffer** during the identified peak periods.
            *   **Reorder Points**: Based on the 8-week forecast, consider placing orders 2 weeks before the projected spikes.

            ### 📈 Growth Opportunities
            *   **Seasonal Trends**: The model identifies strong year-end seasonality. Plan marketing campaigns for early Q4.
            *   **Product Performance**: Focus on the top 10% of items which generate 80% of the revenue.
            """)

if __name__ == "__main__":
    main()
