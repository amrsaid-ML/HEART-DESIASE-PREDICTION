# Heart Disease Prediction Project

## Project Overview
This project uses machine learning to predict heart disease risk based on a set of health metrics. The process involves data preprocessing, feature selection, and training various supervised and unsupervised learning models. The final model is deployed via a Streamlit web application.

## Files and Directories
- `01_data_preprocessing.ipynb`: Notebook for data cleaning and initial preparation.
- `02_pca_analysis.ipynb`: Notebook for Principal Component Analysis.
- `03_feature_selection.ipynb`: Notebook for selecting the most relevant features.
- `04_supervised_learning.ipynb`: Notebook for training and evaluating baseline models.
- `05_unsupervised_learning.ipynb`: Notebook for K-Means and Hierarchical clustering.
- `06_hyperparameter_tuning.ipynb`: Notebook for optimizing model performance and exporting the final model.
- `data/`: Contains the raw and preprocessed datasets.
- `models/`: Contains the final trained model pipeline (`random_forest_pipeline.pkl`).
- `app.py`: The source code for the Streamlit web application.
- `requirements.txt`: Lists all Python dependencies.
