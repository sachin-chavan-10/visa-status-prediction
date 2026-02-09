import os
import sys
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# --- Path Handling ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Check if paths exist
if not os.path.exists(MODELS_PATH):
    print(f"ERROR: 'models' folder not found at {MODELS_PATH}")
    sys.exit(1)

# --- Load Models (Global Scope) ---
try:
    status_model = joblib.load(os.path.join(MODELS_PATH, 'status_case_model_gb.pkl'))
    time_model = joblib.load(os.path.join(MODELS_PATH, 'pt_model.pkl'))
    print("--- Models Loaded Successfully! ---")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Model file missing. {e}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# --- Feature Lists ---
STATUS_FEATURES = [
    'education_of_employee_High School', 'education_of_employee_Master', 
    'education_of_employee_Doctorate', 'has_job_experience', 'requires_job_training', 
    'no_of_employees', 'full_time_position', 'company_age', 'annual_wage', 
    'continent_Asia', 'continent_Europe', 'continent_North America', 
    'continent_Oceania', 'continent_South America', 'region_of_employment_South', 
    'region_of_employment_West'
]

TIME_FEATURES = [
    'has_previous_rejection', 'visa_O-1', 'application_month',
    'annual_wage', 'visa_H-1B', 'no_of_employees', 'visa_L-1', 'visa_TN']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    def get_val(key):
        val = data.get(key, 0)
        try:
            return float(val) if val != "" else 0.0
        except:
            return 0.0

    try:
        status_input = np.array([get_val(f) for f in STATUS_FEATURES]).reshape(1, -1)
        time_input = np.array([get_val(f) for f in TIME_FEATURES]).reshape(1, -1)

        status_pred = status_model.predict(status_input)[0]
        
        try:
            probabilities = status_model.predict_proba(status_input)[0]
            confidence = round(np.max(probabilities) * 100, 2)
        except:
            confidence = 85.0

        time_pred = time_model.predict(time_input)[0]
        MAE_VALUE = 3.5
        range_text = f"{max(0, int(time_pred - MAE_VALUE))} - {int(time_pred + MAE_VALUE)} days"

        return render_template('result.html', 
                               status="Certified" if status_pred == 1 else "Denied", 
                               confidence=confidence,
                               time=f"{int(time_pred)} Days",
                               range_text=range_text)
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)