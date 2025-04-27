import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import mlflow

# ----------------------------------------
# Set local experiment tracking directory
# ----------------------------------------
mlflow.set_experiment("streamlit_predictions")

# Specify the tracking URI to store logs locally
tracking_uri = "file:///C:/Users/dslaa/Documents/GitHub/Insurance Premium/Insurance/mlruns"
mlflow.set_tracking_uri(tracking_uri)
# ----------------------------------------
# Load trained model
# ----------------------------------------
pipeline = joblib.load('RandomForest_model_mlflow.pkl')

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("🏥 Insurance Charges Prediction App")

# Get user inputs
age = st.number_input("Age", min_value=0, max_value=120, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
Claim_Amount = st.number_input("Claim Amount", min_value=0.0, step=0.1)
past_consultations = st.number_input("Past Consultations", min_value=0, step=1)
num_of_steps = st.number_input("Number of Steps", min_value=0, step=1)
Hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, step=0.1)
NUmber_of_past_hospitalizations = st.number_input("Past Hospitalizations", min_value=0, step=1)
Anual_Salary = st.number_input("Annual Salary", min_value=0.0, step=0.1)

sex = st.selectbox("Sex", ['male', 'female'])
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'northwest', 'northeast', 'southeast'])

# Derived features
High_Medical_Dependency = 1 if (Anual_Salary > 0 and Hospital_expenditure / Anual_Salary > 0.3) else 0
Sedentary_Lifestyle_Flag = 1 if num_of_steps < 5000 else 0

# ----------------------------------------
# Make Prediction & Track with MLflow
# ----------------------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[age, bmi, children, Claim_Amount, past_consultations, num_of_steps,
                                Hospital_expenditure, NUmber_of_past_hospitalizations, Anual_Salary,
                                sex, smoker, region, High_Medical_Dependency, Sedentary_Lifestyle_Flag]],
                              columns=['age', 'bmi', 'children', 'Claim_Amount', 'past_consultations',
                                       'num_of_steps', 'Hospital_expenditure', 'NUmber_of_past_hospitalizations',
                                       'Anual_Salary', 'sex', 'smoker', 'region',
                                       'High_Medical_Dependency', 'Sedentary_Lifestyle_Flag'])

    with mlflow.start_run(run_name="Streamlit_Prediction"):
        # Log input parameters to MLflow
        for col in input_data.columns:
            mlflow.log_param(col, input_data[col].values[0])

        # Make prediction
        prediction = pipeline.predict(input_data)

        # Log the predicted value
        mlflow.log_metric("predicted_charge", prediction[0])

        # Show the result to the user
        st.success(f"💰 Predicted Charges: ${prediction[0]:,.2f}")
