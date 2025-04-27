import streamlit as st
import joblib
import pandas as pd
import numpy as np
import mlflow

mlflow.set_experiment("streamlit_predictions")
tracking_uri = "file:///C:/Users/dslaa/Documents/GitHub/Insurance Premium/Insurance/mlruns"
mlflow.set_tracking_uri(tracking_uri)

pipeline = joblib.load('RandomForest_model_mlflow.pkl')

st.title("🏥 Insurance Charges Prediction App")

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

High_Medical_Dependency = 1 if (Anual_Salary > 0 and Hospital_expenditure / Anual_Salary > 0.3) else 0
Sedentary_Lifestyle_Flag = 1 if num_of_steps < 5000 else 0

if st.button("Predict"):
    input_data = pd.DataFrame([[age, bmi, children, Claim_Amount, past_consultations, num_of_steps,
                                Hospital_expenditure, NUmber_of_past_hospitalizations, Anual_Salary,
                                sex, smoker, region, High_Medical_Dependency, Sedentary_Lifestyle_Flag]],
                              columns=['age', 'bmi', 'children', 'Claim_Amount', 'past_consultations',
                                       'num_of_steps', 'Hospital_expenditure', 'NUmber_of_past_hospitalizations',
                                       'Anual_Salary', 'sex', 'smoker', 'region',
                                       'High_Medical_Dependency', 'Sedentary_Lifestyle_Flag'])

    with mlflow.start_run(run_name="Streamlit_Prediction"):
        for col in input_data.columns:
            mlflow.log_param(col, input_data[col].values[0])

        prediction = pipeline.predict(input_data)
        mlflow.log_metric("predicted_charge", prediction[0])
        st.success(f"💰 Predicted Charges: ${prediction[0]:,.2f}")

if st.button("Evaluate on Ground Truth Data"):
    data = {
        'age': [45, 32, 55, 29, 40, 61, 38, 47, 50, 26],
        'bmi': [28.7, 23.5, 31.1, 22.0, 27.8, 35.6, 29.4, 33.2, 26.9, 21.5],
        'children': [2, 0, 3, 1, 2, 0, 2, 1, 4, 0],
        'Claim_Amount': [5000, 2500, 10000, 3000, 4500, 8000, 4000, 6000, 9500, 2000],
        'past_consultations': [3, 1, 4, 2, 3, 5, 2, 3, 6, 1],
        'num_of_steps': [3000, 8000, 2000, 9000, 4000, 1500, 7000, 2500, 1000, 11000],
        'Hospital_expenditure': [15000, 3000, 25000, 5000, 12000, 30000, 10000, 18000, 27000, 2000],
        'NUmber_of_past_hospitalizations': [1, 0, 2, 0, 1, 3, 1, 1, 2, 0],
        'Anual_Salary': [70000, 45000, 100000, 38000, 65000, 120000, 56000, 85000, 95000, 36000],
        'sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'smoker': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no'],
        'region': ['southeast', 'northeast', 'southwest', 'northwest', 'southeast', 'southwest', 'northwest', 'northeast', 'southeast', 'southwest'],
        'actual_charges': [32200, 12450, 42000, 8200, 16500, 51200, 15800, 21000, 38900, 7200]
    }

    df = pd.DataFrame(data)
    df["High_Medical_Dependency"] = df.apply(
        lambda row: 1 if (row["Anual_Salary"] > 0 and row["Hospital_expenditure"] / row["Anual_Salary"] > 0.3) else 0,
        axis=1
    )
    df["Sedentary_Lifestyle_Flag"] = df["num_of_steps"].apply(lambda x: 1 if x < 5000 else 0)

    features = ['age', 'bmi', 'children', 'Claim_Amount', 'past_consultations',
                'num_of_steps', 'Hospital_expenditure', 'NUmber_of_past_hospitalizations',
                'Anual_Salary', 'sex', 'smoker', 'region',
                'High_Medical_Dependency', 'Sedentary_Lifestyle_Flag']

    X = df[features]
    y_true = df["actual_charges"]

    with mlflow.start_run(run_name="Ground_Truth_Evaluation"):
        mlflow.log_param("GroundTruth", True)

        y_pred = pipeline.predict(X)
        df["predicted_charges"] = y_pred
        df["error"] = df["predicted_charges"] - df["actual_charges"]

        mlflow.log_metric("MAE", np.mean(np.abs(df["error"])))
        mlflow.log_metric("MSE", np.mean(df["error"] ** 2))
        mlflow.log_metric("RMSE", np.sqrt(np.mean(df["error"] ** 2)))

        st.subheader("📊 Prediction vs Actual")
        st.dataframe(df[["age", "bmi", "sex", "smoker", "region", "actual_charges", "predicted_charges", "error"]])
