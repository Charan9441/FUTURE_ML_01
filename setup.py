from setuptools import setup, find_packages

setup(
    name="sales_forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "openpyxl",
        "joblib",
        "streamlit"
    ],
    author="Charan",
    description="A modular sales and demand forecasting pipeline by Charan (ML Intern).",
    python_requires=">=3.8",
)
