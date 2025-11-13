import pandas as pd
import joblib
import os

# Load saved artifacts
clf = joblib.load("rf_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
label_encoder = joblib.load("label_encoder.joblib")

def predict(input_data: dict):
    """
    input_data: dict of feature_name: value
    Returns: predicted Reason as string
    """
    df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(df)
    y_pred = clf.predict(X_processed)
    return label_encoder.inverse_transform(y_pred)[0]

