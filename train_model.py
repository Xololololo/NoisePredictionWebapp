# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:58:47 2025

@author: hui_j
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib  # for saving/loading model

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\hui_j\OneDrive\Documents\Eng Neo Noise Prediction Webapp\Prediction_Webapp\Jan to Sep Eng Neo Forest Noise dataset.csv")

# Drop unnecessary columns
df = df.drop(['Date', 'Status'], axis=1, errors='ignore')

# -------------------------------
# 2. Define Features and Target
# -------------------------------
num_var = [var for var in df.columns if df[var].dtype != 'O' and var != 'Reason']
cat_var = [var for var in df.columns if df[var].dtype == 'O' and var != 'Reason']
target_var = 'Reason'

X = df.drop(target_var, axis=1)
y = df[target_var]

# -------------------------------
# 3. Encode Target
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# 4. Preprocess Features
# -------------------------------
ct = ColumnTransformer(
    transformers=[
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_var),
        ('num_passthrough', 'passthrough', num_var)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)
X_processed = ct.fit_transform(X)
feature_names = list(ct.get_feature_names_out())

# -------------------------------
# 5. Split Train/Test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# 6. Train Random Forest Model
# -------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
clf.fit(X_train, y_train)

# -------------------------------
# 7. Save Model and Preprocessor
# -------------------------------
import os
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rf_model.joblib")
joblib.dump(ct, "models/preprocessor.joblib")
joblib.dump(le, "models/label_encoder.joblib")

# Model, preprocessor, and encoder can now be loaded in a web app for predictions
