import joblib
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


model = joblib.load("RandomForest_model.pkl")

def test_model_prediction():
    sample_input = pd.DataFrame([{
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
    }])

    prediction = model.predict(sample_input)

    assert isinstance(prediction[0], (float, np.floating)), "Prediction is not a float"
    assert prediction.shape == (1,), "Prediction should return exactly one value"
