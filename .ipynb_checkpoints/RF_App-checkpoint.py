import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Insurance Charges Prediction")

pipeline = joblib.load('RandomForest_model.pkl')

age = st.number_input("Age", min_value=0, max_value=120, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, step=1)
Claim_Amount = st.number_input("Claim Amount", min_value=0.0, step=0.1)
past_consultations = st.number_input("Past Consultations", min_value=0, step=1)
num_of_steps = st.number_input("Number of Steps", min_value=0, step=1)
Hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, step=0.1)
NUmber_of_past_hospitalizations = st.number_input("Number of Past Hospitalizations", min_value=0, step=1)
Anual_Salary = st.number_input("Annual Salary", min_value=0.0, step=0.1)

sex = st.selectbox("Sex", ['male', 'female'])
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'northwest', 'northeast', 'southeast'])

High_Medical_Dependency = 1 if (Hospital_expenditure / Anual_Salary) > 0.3 else 0
Sedentary_Lifestyle_Flag = 1 if num_of_steps < 5000 else 0

if st.button("Predict"):
    input_data = pd.DataFrame([[age, bmi, children, Claim_Amount, past_consultations, num_of_steps,
                                Hospital_expenditure, NUmber_of_past_hospitalizations, Anual_Salary,
                                sex, smoker, region, High_Medical_Dependency, Sedentary_Lifestyle_Flag]],
                              columns=['age', 'bmi', 'children', 'Claim_Amount', 'past_consultations',
                                       'num_of_steps', 'Hospital_expenditure', 'NUmber_of_past_hospitalizations',
                                       'Anual_Salary', 'sex', 'smoker', 'region', 'High_Medical_Dependency',
                                       'Sedentary_Lifestyle_Flag'])

    prediction = pipeline.predict(input_data)

    st.write(f"Predicted Charges: ${prediction[0]:,.2f}")
