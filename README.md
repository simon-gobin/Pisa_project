
# PISA Project

This repository contains a complete pipeline for the extraction, transformation, loading, and analysis of data from the **Programme for International Student Assessment (PISA)**. The project integrates large-scale educational data into a structured database environment and applies advanced machine learning techniques to perform predictive analytics.

## Overview

The project includes:
- **ETL Pipeline:** Cleans, preprocesses, and transforms raw PISA datasets.
- **Database Integration:** Loads processed data into an Azure SQL database for scalable access and management.
- **Machine Learning Benchmarking:** Trains and evaluates multiple machine learning models to predict outcomes based on student, teacher, and school-level features.
- **Optimized Data Storage:** Saves cleaned datasets in Parquet format to enhance performance in data loading and model training workflows.

## Project Structure

- `ETL_class/`: Classes for data extraction, cleaning, and transformation.
- `azure_module/`: Modules for database schema creation and data insertion into Azure SQL.
- `bench_mark_module/`: Machine learning benchmarking tools for model evaluation.
- `data_loader/`: Scripts for efficient loading of preprocessed data.
- `ETL_run.py`: Executes the local ETL process.
- `azure_load_run.py`: Uploads processed datasets to the Azure database.
- `load_parquet_files.py`: Loads Parquet files for further modeling.
- `target_vallue_normalized_and_load.py`: Normalizes target variables to prepare them for analysis.
- `requirements.txt`: Lists the required Python packages.

## Key Features

- **Data Engineering:** Automated ETL process for large-scale educational datasets.
- **Cloud Integration:** Azure SQL database integration for centralized data management.
- **Model Development:** Machine learning pipeline supporting XGBoost, Random Forest, ElasticNet, SVR, and more.
- **Performance Benchmarking:** Evaluation metrics include R², MAE, MSE, and MAPE, with optional training time penalization.
- **Scalability:** Use of Parquet files for efficient data storage and access.

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- PyODBC, SQLAlchemy
- Azure SQL Database
- Matplotlib

## Setup Instructions

1. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the ETL pipeline:
   ```bash
   python ETL_run.py
   ```

3. (Optional) Upload processed data to Azure:
   ```bash
   python azure_load_run.py
   ```

4. Load datasets for modeling:
   ```bash
   python load_parquet_files.py
   ```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Author

**Simon Gobin**  
[GitHub Profile](https://github.com/simon-gobin)

---

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
