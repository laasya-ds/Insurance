import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Title for the web app
st.title("Insurance Charges Prediction")

# Load the trained decision tree model and scaler from the .pkl files
pipeline = joblib.load('DT_model_region.pkl')  # Load the entire pipeline (model + scaler + encoder)

# Define input fields for user input
age = st.number_input("Age", min_value=0, max_value=120, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, step=1)

# Define input field for 'region' (which needs to be one-hot encoded)
region = st.selectbox("Region", ['southwest', 'northwest', 'northeast', 'southeast'])

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Prepare the input features as a DataFrame (ensure it has the correct column names)
    input_data = pd.DataFrame([[age, bmi, children, region]], columns=['age', 'bmi', 'children', 'region'])

    # Use the pipeline to make predictions (it handles scaling and encoding)
    prediction = pipeline.predict(input_data)

    # Display the prediction result
    st.write(f"Predicted Charges: ${prediction[0]:,.2f}")
