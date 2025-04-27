import joblib
import pandas as pd
import numpy as np
import os
import pytest

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the model
model = joblib.load("RandomForest_model_mlflow.pkl")

# Helper function
def sedentary_lifestyle_flag(num_of_steps):
    return int(num_of_steps < 5000)

def test_model_saved():
    assert os.path.exists('RandomForest_model_mlflow.pkl'), "Model file not found"

@pytest.mark.parametrize(
    "num_of_steps, expected_flag",
    [
        (0, 1),
        (4999, 1),
        (5000, 0),
        (10000, 1),
    ]
)
def test_sedentary_lifestyle(num_of_steps, expected_flag):
    assert sedentary_lifestyle_flag(num_of_steps) == expected_flag

@pytest.mark.parametrize(
    "input_data",
    [
        {
            'age': 35,
            'bmi': 28.0,
            'children': 2,
            'Claim_Amount': 10000,
            'past_consultations': 5,
            'num_of_steps': 4000,
            'Hospital_expenditure': 30000,
            'NUmber_of_past_hospitalizations': 1,
            'Anual_Salary': 90000,
            'High_Medical_Dependency': 1,
            'Sedentary_Lifestyle_Flag': 1,
            'sex': 'female',
            'smoker': 'no',
            'region': 'northeast'
        },
        {
            'age': 50,
            'bmi': 32.5,
            'children': 3,
            'Claim_Amount': 15000,
            'past_consultations': 2,
            'num_of_steps': 6000,
            'Hospital_expenditure': 40000,
            'NUmber_of_past_hospitalizations': 2,
            'Anual_Salary': 120000,
            'High_Medical_Dependency': 0,
            'Sedentary_Lifestyle_Flag': 0,
            'sex': 'male',
            'smoker': 'yes',
            'region': 'southwest'
        },
        {
            'age': 28,
            'bmi': 23.0,
            'children': 0,
            'Claim_Amount': 5000,
            'past_consultations': 1,
            'num_of_steps': 8000,
            'Hospital_expenditure': 10000,
            'NUmber_of_past_hospitalizations': 0,
            'Anual_Salary': 70000,
            'High_Medical_Dependency': 0,
            'Sedentary_Lifestyle_Flag': 0,
            'sex': 'female',
            'smoker': 'no',
            'region': 'northwest'
        }
    ]
)
def test_model_prediction(input_data):
    sample_input = pd.DataFrame([input_data])

    prediction = model.predict(sample_input)

    assert isinstance(prediction[0], (float, np.floating)), "Prediction is not a float"
    assert prediction.shape == (1,), "Prediction should return exactly one value"
