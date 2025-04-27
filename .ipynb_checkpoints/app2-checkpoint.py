import streamlit as st
import joblib
import numpy as np

# Title for the web app
st.title("Insurance Charges Prediction - 1")

# Load the trained decision tree model and scaler from the .pkl files
model = joblib.load('DT_model_2.pkl')  # Update the path if needed (relative or absolute path)
scaler = joblib.load('scaler_model2.pkl')  # Load the saved scaler

# Define input fields for user input
age = st.number_input("Age", min_value=0, max_value=120, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, step=1)

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Prepare the input features as a 2D numpy array (model expects 2D array)
    input_features = np.array([[age, bmi, children]])
    
    # Scale the input features using the loaded scaler
    input_features_scaled = scaler.transform(input_features)  # Apply scaling
    
    # Make the prediction using the trained model
    prediction = model.predict(input_features_scaled)
    
    # Display the prediction result
    st.write(f"Predicted Charges: ${prediction[0]:,.2f}")
