Heart Disease Prediction Project

[Project Overview]:

This project aims to analyze, predict, and visualize heart disease risks using machine
learning. The workflow involves data preprocessing, feature selection, dimensionality
reduction (PCA), model training, evaluation, and deployment. Classification models like
Logistic Regression, Decision Trees, Random Forest, and SVM will be used, alongside
K-Means and Hierarchical Clustering for unsupervised learning. Additionally, a Streamlit UI
will be built for user interaction, deployed via Ngrok, and the project will be hosted on GitHub



[Files and Directories]:

- `01_data_preprocessing.ipynb`: Notebook for data cleaning and initial preparation.
- `02_pca_analysis.ipynb`: Notebook for Principal Component Analysis.
- `03_feature_selection.ipynb`: Notebook for selecting the most relevant features.
- `04_supervised_learning.ipynb`: Notebook for training and evaluating baseline models.
- `05_unsupervised_learning.ipynb`: Notebook for K-Means and Hierarchical clustering.
- `06_hyperparameter_tuning.ipynb`: Notebook for optimizing model performance and exporting the final model.
- `data/`: Contains the raw and preprocessed datasets.
- `models/`: Contains the final trained model pipeline (`final_dataset.pkl`).
- `app.py`: The source code for the Streamlit web application.
- `requirements.txt`: Lists all Python dependencies.