from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -------------------------------
# 1. Safely locate models folder
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # The path where Webapp.py is located
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Debug: print paths (optional, remove in production)
print("🔍 Base directory:", BASE_DIR)
print("🔍 Model directory:", MODEL_DIR)
print("🔍 Files in model directory:", os.listdir(MODEL_DIR))

# Load models using absolute paths
clf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
ct = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

# -------------------------------
# 2. Home route with prediction
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            # --- Optional validation for SensorID ---
            sensor = request.form.get("SensorID")
            if sensor not in ["EN-N01", "EN-N02", "EN-N03"]:
                error = f"Invalid SensorID: {sensor}"
                return render_template("index.html", prediction=None, confidence=None, error=error)

            # --- Prepare input data for prediction ---
            data = {
                "SensorID": [sensor],                 # now a string
                "Period": [request.form.get("Period")],
                "Value": [float(request.form.get("Value"))]
            }

            df = pd.DataFrame(data)

            # --- Transform and predict ---
            X_proc = ct.transform(df)
            pred_encoded = clf.predict(X_proc)
            prediction = le.inverse_transform(pred_encoded)[0]
            confidence = round(clf.predict_proba(X_proc).max(), 3)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, confidence=confidence, error=error)

# -------------------------------
# 3. Render-specific adjustments
# -------------------------------
if __name__ == "__main__":
    # Render assigns a PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    # Must set host to 0.0.0.0 so it’s accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)
