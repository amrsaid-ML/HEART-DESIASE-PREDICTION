import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
import plotly.express as px

# first i will  Load the Model and Data with error handling

# Checking if the data file exists
data_path = os.path.join('final_dataset.csv')
if not os.path.exists(data_path):
    st.error(f"File not found: {data_path}")
    st.stop()

# Checking if the model file exists
model_path = os.path.join( 'final_model.pkl')
if not os.path.exists(model_path):
    st.error(f"File not found: {model_path}")
    st.stop()


# Loading the files
df_final = pd.read_csv(data_path)
pipeline = joblib.load(model_path)


# Get feature names from the loaded data
model_features = df_final.drop('target', axis=1).columns.tolist()


# now we start using Streamlit UI to create the app
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Prediction App ")
st.markdown("Enter the patient's health data to predict their heart disease status.")


# now setting a Prediction UI 
st.header("Predict Heart Disease Risk")
col1, col2, col3 = st.columns(3)


with col1:
    age = st.slider('Age', 29, 77, 50)
    chol = st.number_input('Cholesterol (mg/dl)', 126, 564, 250)
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 94, 200, 130)

with col2:
    oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    thalch = st.number_input('Max Heart Rate (thalch)', 71, 202, 150)
    ca = st.selectbox('Number of Major Vessels Colored (ca)', options=[0, 1, 2, 3])

with col3:
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    cp = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'])
    exang = st.selectbox('Exercise Induced Angina', options=['No', 'Yes'])
    thal = st.selectbox('Thallium Stress Test Result', options=['Normal', 'Fixed Defect', 'Reversible Defect'])



# now i will define the logic to prediction

#Prediction Logic
def get_prediction_data():
    data = {}
    data['sex_Male'] = 1 if sex == 'Male' else 0
    data['cp_atypical angina'] = 1 if cp == 'Atypical Angina' else 0
    data['cp_non-anginal'] = 1 if cp == 'Non-Anginal' else 0
    data['exang_True'] = 1 if exang == 'Yes' else 0
    data['thal_normal'] = 1 if thal == 'Normal' else 0
    data['thal_reversable defect'] = 1 if thal == 'Reversible Defect' else 0

    data['age'] = age
    data['chol'] = chol
    data['trestbps'] = trestbps
    data['oldpeak'] = oldpeak
    data['thalch'] = thalch
    data['ca'] = ca

    return pd.DataFrame([data])


if st.button('Predict'):
    with st.spinner('Making a prediction...'):
        user_data = get_prediction_data()
        
        # Ensuring all columns are present and in the correct order
        user_data_reordered = user_data.reindex(columns=model_features, fill_value=0)

        prediction = pipeline.predict(user_data_reordered)[0]
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("The model predicts the patient has a **HIGH** risk of heart disease.")
        else:
            st.success("The model predicts the patient has a **LOW** risk of heart disease.")




#  now we start the visualization
# Data Visualization Section

st.header("Data Visualization")
st.markdown("Explore trends in the dataset used to train the model.")

# for example Age Distribution
st.subheader("Age Distribution of Patients")
age_fig = px.histogram(df_final, x='age', color='target', 
                       title='Age vs. Heart Disease Status',
                       labels={'target': 'Heart Disease'},
                       color_discrete_map={0: 'blue', 1: 'red'})
st.plotly_chart(age_fig, use_container_width=True)

# and Heart Disease by Sex
st.subheader("Heart Disease by Sex")
sex_data = df_final.copy()
sex_data['sex'] = sex_data['sex_Male'].map({0: 'Female', 1: 'Male'})
sex_fig = px.bar(sex_data, x='sex', color='target',
                  title='Heart Disease Cases by Sex',
                  labels={'target': 'Heart Disease'},
                  color_discrete_map={0: 'blue', 1: 'red'})
st.plotly_chart(sex_fig, use_container_width=True)



    
