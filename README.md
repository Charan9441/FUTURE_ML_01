# 📈 Sales & Demand Forecasting System

A professional, modular, and deployment-ready Machine Learning pipeline designed for business demand forecasting. This project transforms historical transactional data into actionable insights and future projections.

**Live demo:** https://sdfsystem.streamlit.app/

## 🚀 Key Features
- **Modular Architecture**: Reusable logic separated into dedicated Python modules.
- **Interactive Narrative**: Professional Jupyter Notebook for data storytelling and EDA.
- **Deployment-Ready**: Includes a Streamlit dashboard starter for cloud deployment.
- **Live Deployment**: A published dashboard is available online for stakeholders to review forecasts.
- **Robust Feature Engineering**: Leverages time-series lags and rolling averages to capture trends.
- **Cross-Platform**: Uses `pathlib` for compatibility across Windows, Linux, and Google Colab.

## 📂 Folder Structure
```
project/
│
├── data/               # Raw and processed datasets
│   └── Online Retail.xlsx
│
├── notebooks/          # Documentation and EDA
│   └── sales_forecasting.ipynb
│
├── src/                # Modular source code
│   ├── data_loader.py  # Dataset loading utilities
│   ├── preprocessing.py # Cleaning and feature engineering
│   ├── visualization.py # Professional business charts
│   ├── forecasting.py   # Model training and evaluation
│   └── utils.py         # Path and directory management
│
├── models/             # Saved model binaries (.pkl)
│
├── outputs/            # Generated assets
│   ├── charts/         # PNG visualizations
│   └── predictions/    # CSV forecast results
│
├── app/                # Web application
│   └── app.py          # Streamlit dashboard starter
│
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
├── .gitignore          # Version control exclusions
└── setup.py            # Package installation script
```

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
# Clone the repository
git clone <repository-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `notebooks/sales_forecasting.ipynb` in VS Code or Jupyter Lab to view the full analysis and training process.

### 3. Run the Streamlit app locally
```bash
streamlit run app/app.py
```

### 4. Live Deployment
A deployed version of the dashboard is available at: https://sdfsystem.streamlit.app/

### 5. Running on Google Colab
1. Upload the `notebooks/sales_forecasting.ipynb` to Google Colab.
2. Upload the `Online Retail.xlsx` file to the `/content/` directory.
3. Upload the `src/` folder to the `/content/` directory to enable modular imports.
4. Run the cells sequentially.

## 📊 Insights & Strategy
- **Inventory Optimization**: Use the 8-week forecast to adjust stock levels ahead of the Q4 holiday surge.
- **Market Focus**: Target the UK market for localized promotions while scaling Germany and France.
- **Scalability**: The modular `src/` files can be imported into any Flask/FastAPI backend for production deployment.

---
**Author**: Charan (Data Scientist ML Intern)
**License**: MIT
