# import os
# import json
# import sys
# import pandas as pd
# import numpy as np
# import joblib
# from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # --- Database Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'visa_records.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# class VisaApplication(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     annual_wage = db.Column(db.Float)
#     visa_type = db.Column(db.String(20))
#     education = db.Column(db.String(50))
#     status_prediction = db.Column(db.String(20))
#     time_prediction = db.Column(db.Integer)
#     confidence = db.Column(db.Float)

# # --- Model Loading ---
# MODELS_PATH = os.path.join(BASE_DIR, 'models')
# try:
#     status_model = joblib.load(os.path.join(MODELS_PATH, 'status_case_model_gb.pkl'))
#     time_model = joblib.load(os.path.join(MODELS_PATH, 'pt_model.pkl'))
#     print("--- Models Loaded Successfully! ---")
# except Exception as e:
#     print(f"CRITICAL ERROR: {e}")
#     sys.exit(1)

# # These lists must match the features the model was trained on
# STATUS_FEATURES = [
#     'education_of_employee_High School', 'education_of_employee_Master', 
#     'education_of_employee_Doctorate', 'has_job_experience', 'requires_job_training', 
#     'no_of_employees', 'full_time_position', 'company_age', 'annual_wage', 
#     'continent_Asia', 'continent_Europe', 'continent_North America', 
#     'continent_Oceania', 'continent_South America', 'region_of_employment_South', 
#     'region_of_employment_West'
# ]

# TIME_FEATURES = [
#     'has_previous_rejection', 'visa_O-1', 'application_month',
#     'annual_wage', 'visa_H-1B', 'no_of_employees', 'visa_L-1', 'visa_TN'
# ]

# @app.route('/')
# def home():
#     # Landing Page
#     return render_template('index.html')

# @app.route('/assessment')
# def assessment():
#     # Form Page
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
    
#     # Improved value fetcher to handle empty strings and missing keys
#     def get_val(key):
#         val = data.get(key, 0)
#         try:
#             return float(val) if val != "" else 0.0
#         except:
#             return 0.0

#     try:
#         # 1. Prepare Inputs as DataFrames
#         status_input_values = [get_val(f) for f in STATUS_FEATURES]
#         status_df = pd.DataFrame([status_input_values], columns=STATUS_FEATURES)
        
#         time_input_values = [get_val(f) for f in TIME_FEATURES]
#         time_df = pd.DataFrame([time_input_values], columns=TIME_FEATURES)

#         # 2. Perform Predictions
#         status_pred = status_model.predict(status_df)[0]
#         time_raw_pred = time_model.predict(time_df)[0]
#         status_label = "CERTIFIED" if status_pred == 1 else "DENIED"

#         # 3. Confidence Calculation
#         try:
#             probabilities = status_model.predict_proba(status_df)[0]
#             confidence = round(np.max(probabilities) * 100, 2)
#         except:
#             confidence = 85.0

#         # 4. Feature Importance Logic for Charts
#         status_global_imp = status_model.feature_importances_
#         user_status_imp = []
#         for i, feature in enumerate(STATUS_FEATURES):
#             if get_val(feature) > 0:
#                 user_status_imp.append({
#                     "feature": feature.replace('_', ' ').replace('education of employee', '').replace('continent', 'Origin:').title(),
#                     "raw_weight": float(status_global_imp[i])
#                 })
        
#         status_total = sum(item['raw_weight'] for item in user_status_imp) or 1.0
#         for item in user_status_imp:
#             item['value'] = item['raw_weight'] / status_total
#         user_status_imp = sorted(user_status_imp, key=lambda x: x['value'], reverse=True)

#         # 5. Time Sensitivity
#         user_time_imp = []
#         base_time_pred = time_raw_pred
#         for i, feature in enumerate(TIME_FEATURES):
#             if get_val(feature) > 0:
#                 sensitivity_df = time_df.copy()
#                 sensitivity_df.iloc[0, i] = 0 
#                 alt_pred = time_model.predict(sensitivity_df)[0]
#                 impact = abs(base_time_pred - alt_pred)
#                 user_time_imp.append({
#                     "feature": feature.replace('_', ' ').title(),
#                     "impact_score": float(impact)
#                 })

#         time_total_impact = sum(item['impact_score'] for item in user_time_imp) or 1.0
#         for item in user_time_imp:
#             item['value'] = item['impact_score'] / time_total_impact
#         user_time_imp = sorted(user_time_imp, key=lambda x: x['value'], reverse=True)

#         # 6. Final Outputs
#         final_days = int(time_raw_pred)
#         time_range = f"{final_days - 3} - {final_days + 3}"

#         # Save to Database
#         new_record = VisaApplication(
#             annual_wage=get_val('annual_wage'),
#             visa_type=data.get('visa_category', 'Other'),
#             education=data.get('education_of_employee_Master', 'Degree'), # Basic mapping
#             status_prediction=status_label,
#             time_prediction=final_days,
#             confidence=confidence
#         )
#         db.session.add(new_record)
#         db.session.commit()

#         return render_template('result.html', 
#                                status=status_label, 
#                                confidence=confidence,
#                                days=final_days,
#                                time_range=time_range,
#                                status_imp=json.dumps(user_status_imp),
#                                time_imp=json.dumps(user_time_imp),
#                                est_date=(datetime.now() + timedelta(days=final_days)).strftime('%B %d, %Y'))

#     except Exception as e:
#         # Printing the error to the console helps you debug
#         print(f"Error during prediction: {e}")
#         return f"Prediction Error: {str(e)}"

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)


# import os
# import json
# import sys
# import pandas as pd
# import numpy as np
# import joblib
# from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # --- Database Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'visa_records.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# class VisaApplication(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     annual_wage = db.Column(db.Float)
#     visa_type = db.Column(db.String(20))
#     education = db.Column(db.String(50))
#     status_prediction = db.Column(db.String(20))
#     time_prediction = db.Column(db.Integer)
#     confidence = db.Column(db.Float)

# # --- Model Loading ---
# MODELS_PATH = os.path.join(BASE_DIR, 'models')
# try:
#     # Using joblib for Gradient Boosting models
#     status_model = joblib.load(os.path.join(MODELS_PATH, 'status_case_model_gb.pkl'))
#     time_model = joblib.load(os.path.join(MODELS_PATH, 'pt_model.pkl'))
#     print("--- Models Loaded Successfully! ---")
# except Exception as e:
#     print(f"CRITICAL ERROR: {e}")
#     sys.exit(1)

# # Feature lists mapped exactly to model training columns
# STATUS_FEATURES = [
#     'education_of_employee_High School', 'education_of_employee_Master', 
#     'education_of_employee_Doctorate', 'has_job_experience', 'requires_job_training', 
#     'no_of_employees', 'full_time_position', 'company_age', 'annual_wage', 
#     'continent_Asia', 'continent_Europe', 'continent_North America', 
#     'continent_Oceania', 'continent_South America', 'region_of_employment_South', 
#     'region_of_employment_West'
# ]

# TIME_FEATURES = [
#     'has_previous_rejection', 'visa_O-1', 'application_month',
#     'annual_wage', 'visa_H-1B', 'no_of_employees', 'visa_L-1', 'visa_TN'
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/assessment')
# def assessment():
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
    
#     def get_val(key):
#         val = data.get(key, 0)
#         try:
#             return float(val) if val != "" else 0.0
#         except:
#             return 0.0

#     try:
#         # 1. Input Preparation
#         status_input_values = [get_val(f) for f in STATUS_FEATURES]
#         status_df = pd.DataFrame([status_input_values], columns=STATUS_FEATURES)
        
#         time_input_values = [get_val(f) for f in TIME_FEATURES]
#         time_df = pd.DataFrame([time_input_values], columns=TIME_FEATURES)

#         # 2. Predictions
#         status_pred = status_model.predict(status_df)[0]
#         time_raw_pred = time_model.predict(time_df)[0]
#         status_label = "CERTIFIED" if status_pred == 1 else "DENIED"

#         # 3. Confidence Logic (Handling models without predict_proba)
#         if hasattr(status_model, "predict_proba"):
#             probabilities = status_model.predict_proba(status_df)[0]
#             confidence = round(np.max(probabilities) * 100, 1)
#         else:
#             confidence = 92.5 # Smart fallback

#         # 4. Global Feature Importance (Status)
#         user_status_imp = []
#         if hasattr(status_model, 'feature_importances_'):
#             importances = status_model.feature_importances_
#             for i, feat in enumerate(STATUS_FEATURES):
#                 if get_val(feat) > 0 or feat in ['annual_wage', 'no_of_employees']:
#                     user_status_imp.append({
#                         "feature": feat.replace('_', ' ').replace('education of employee', '').replace('continent', 'Origin:').title(),
#                         "weight": float(importances[i])
#                     })
        
#         # Normalize weights for the chart
#         s_total = sum(d['weight'] for d in user_status_imp) or 1
#         for d in user_status_imp: d['value'] = d['weight'] / s_total
#         user_status_imp = sorted(user_status_imp, key=lambda x: x['value'], reverse=True)[:5]

#         # 5. Sensitivity Analysis (Time)
#         user_time_imp = []
#         for i, feat in enumerate(TIME_FEATURES):
#             if get_val(feat) > 0:
#                 temp_df = time_df.copy()
#                 temp_df.iloc[0, i] = 0 
#                 impact = abs(time_raw_pred - time_model.predict(temp_df)[0])
#                 user_time_imp.append({
#                     "feature": feat.replace('_', ' ').title(),
#                     "impact": float(impact)
#                 })

#         t_total = sum(d['impact'] for d in user_time_imp) or 1
#         for d in user_time_imp: d['value'] = d['impact'] / t_total
#         user_time_imp = sorted(user_time_imp, key=lambda x: x['value'], reverse=True)[:5]

#         # 6. Database Persistence
#         # Logic to find which education level was selected for the DB
#         edu_selected = "Bachelor"
#         for edu in ['Master', 'Doctorate', 'High School']:
#             if get_val(f'education_of_employee_{edu}') == 1:
#                 edu_selected = edu

#         new_record = VisaApplication(
#             annual_wage=get_val('annual_wage'),
#             visa_type=data.get('visa_category', 'E-2'),
#             education=edu_selected,
#             status_prediction=status_label,
#             time_prediction=int(time_raw_pred),
#             confidence=confidence
#         )
#         db.session.add(new_record)
#         db.session.commit()

#         return render_template('result.html', 
#                                status=status_label, 
#                                confidence=confidence,
#                                days=int(time_raw_pred),
#                                time_range=f"{int(time_raw_pred)-2}-{int(time_raw_pred)+2}",
#                                status_imp=json.dumps(user_status_imp),
#                                time_imp=json.dumps(user_time_imp),
#                                est_date=(datetime.now() + timedelta(days=int(time_raw_pred))).strftime('%b %d, %Y'))

#     except Exception as e:
#         print(f"Prediction Error: {e}")
#         return f"System Error: {str(e)}", 500

    

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, port=5000)

# import os
# import json
# import sys
# import pandas as pd
# import numpy as np
# import joblib
# import google.generativeai as genai
# from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # --- Gemini Configuration ---
# # Replace with your actual API Key or use environment variables
# genai.configure(api_key="AIzaSyDAy99EYuZqSa6IsRL_YkSLYtGTWGHBiGE")
# gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# # --- Database Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'visa_records.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# class VisaApplication(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     annual_wage = db.Column(db.Float)
#     visa_type = db.Column(db.String(20))
#     education = db.Column(db.String(50))
#     status_prediction = db.Column(db.String(20))
#     time_prediction = db.Column(db.Integer)
#     confidence = db.Column(db.Float)

# # --- Model Loading ---
# MODELS_PATH = os.path.join(BASE_DIR, 'models')
# try:
#     status_model = joblib.load(os.path.join(MODELS_PATH, 'status_case_model_gb.pkl'))
#     time_model = joblib.load(os.path.join(MODELS_PATH, 'pt_model.pkl'))
#     print("--- Models Loaded Successfully! ---")
# except Exception as e:
#     print(f"CRITICAL ERROR: {e}")
#     sys.exit(1)

# STATUS_FEATURES = [
#     'education_of_employee_High School', 'education_of_employee_Master', 
#     'education_of_employee_Doctorate', 'has_job_experience', 'requires_job_training', 
#     'no_of_employees', 'full_time_position', 'company_age', 'annual_wage', 
#     'continent_Asia', 'continent_Europe', 'continent_North America', 
#     'continent_Oceania', 'continent_South America', 'region_of_employment_South', 
#     'region_of_employment_West'
# ]

# TIME_FEATURES = [
#     'has_previous_rejection', 'visa_O-1', 'application_month',
#     'annual_wage', 'visa_H-1B', 'no_of_employees', 'visa_L-1', 'visa_TN'
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/assessment')
# def assessment():
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
    
#     def get_val(key):
#         val = data.get(key, 0)
#         try:
#             return float(val) if val != "" else 0.0
#         except:
#             return 0.0

#     try:
#         # 1. Input Preparation
#         status_input_values = [get_val(f) for f in STATUS_FEATURES]
#         status_df = pd.DataFrame([status_input_values], columns=STATUS_FEATURES)
        
#         time_input_values = [get_val(f) for f in TIME_FEATURES]
#         time_df = pd.DataFrame([time_input_values], columns=TIME_FEATURES)

#         # 2. Predictions
#         status_pred = status_model.predict(status_df)[0]
#         time_raw_pred = time_model.predict(time_df)[0]
#         status_label = "CERTIFIED" if status_pred == 1 else "DENIED"

#         # 3. Confidence Logic
#         if hasattr(status_model, "predict_proba"):
#             probabilities = status_model.predict_proba(status_df)[0]
#             confidence = round(np.max(probabilities) * 100, 1)
#         else:
#             confidence = 92.5

#         # 4. Feature Importance (Status)
#         user_status_imp = []
#         if hasattr(status_model, 'feature_importances_'):
#             importances = status_model.feature_importances_
#             for i, feat in enumerate(STATUS_FEATURES):
#                 if get_val(feat) > 0 or feat in ['annual_wage', 'no_of_employees']:
#                     user_status_imp.append({
#                         "feature": feat.replace('_', ' ').replace('education of employee', '').replace('continent', 'Origin:').title(),
#                         "weight": float(importances[i])
#                     })

        
#         s_total = sum(d['weight'] for d in user_status_imp) or 1
#         for d in user_status_imp: d['value'] = d['weight'] / s_total
#         user_status_imp = sorted(user_status_imp, key=lambda x: x['value'], reverse=True)[:10]

#         # 5. Sensitivity Analysis (Time)
#         user_time_imp = []
#         for i, feat in enumerate(TIME_FEATURES):
#             if get_val(feat) > 0:
#                 temp_df = time_df.copy()
#                 temp_df.iloc[0, i] = 0 
#                 impact = abs(time_raw_pred - time_model.predict(temp_df)[0])
#                 user_time_imp.append({
#                     "feature": feat.replace('_', ' ').title(),
#                     "impact": float(impact)
#                 })

#         t_total = sum(d['impact'] for d in user_time_imp) or 1
#         for d in user_time_imp: d['value'] = d['impact'] / t_total
#         user_time_imp = sorted(user_time_imp, key=lambda x: x['value'], reverse=True)[:10]

#         # 6. Database Persistence
#         edu_selected = "Bachelor"
#         for edu in ['Master', 'Doctorate', 'High School']:
#             if get_val(f'education_of_employee_{edu}') == 1:
#                 edu_selected = edu

#         # 7. --- GEMINI INTEGRATION ---
#         # Construct a textual summary of chart data for the AI to "read"
#         status_chart_summary = ", ".join([f"{d['feature']} ({round(d['value']*100)}% impact)" for d in user_status_imp])

#         time_chart_summary = ", ".join([f"{d['feature']} ({round(d['value']*100)}% impact)" for d in user_time_imp])
        
#         prompt = f"""
#         Act as a Senior US Immigration Analyst. 
#         I have a visa prediction with the following data:
#         - Result: {status_label}
#         - Confidence: {confidence}%
#         - Processing Time: {int(time_raw_pred)} days
#         - Visa Type: {data.get('visa_category', 'Specified')}

#         ML Model Insights:
#         1. Factors affecting APPROVAL: {status_chart_summary}
#         2. Factors affecting TIME DELAYS: {time_chart_summary}

#         Provide a 3-short paragraph 'Strategic Analysis':
#         Paragraph 1: Explain why they were {status_label} based on the approval factors.

#         Paragraph 2: Analyze the TIME SENSITIVITY. Specifically explain why {user_time_imp[0]['feature']} is impacting their wait time.
        
#         Paragraph 3: Give a specific 'Pro-Tip' to optimize their application.
        
#         Keep the tone professional and expert. Do not use bullet points.
#         """
        
#         # try:
#         #     response = gemini_model.generate_content(prompt)
#         #     ai_analysis = response.text.strip()
#         # except Exception as gem_e:
#         #     print(f"Gemini Error: {gem_e}")
#         #     ai_analysis = "Our AI consultant is currently analyzing your charts. Based on statistical data, your profile shows a strong correlation with the predicted status due to your professional background."

#         # 8. Save Record
#         new_record = VisaApplication(
#             annual_wage=get_val('annual_wage'),
#             visa_type=data.get('visa_category', 'E-2'),
#             education=edu_selected,
#             status_prediction=status_label,
#             time_prediction=int(time_raw_pred),
#             confidence=confidence
#         )
#         db.session.add(new_record)
#         db.session.commit()

#         return render_template('result.html', 
#                                status=status_label, 
#                                confidence=confidence,
#                                days=int(time_raw_pred),
#                                time_range=f"{int(time_raw_pred)-2}-{int(time_raw_pred)+2}",
#                                status_imp=json.dumps(user_status_imp),
#                                time_imp=json.dumps(user_time_imp),
#                             #    ai_analysis=ai_analysis,
#                                est_date=(datetime.now() + timedelta(days=int(time_raw_pred))).strftime('%b %d, %Y'))

#     except Exception as e:
#         print(f"Prediction Error: {e}")
#         return f"System Error: {str(e)}", 500

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, port=5000)

import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

app = Flask(__name__)
# app.config['SECRET_KEY'] = '87afef454f594ce371244deb853a9e6c4423d1edffa8111c' 
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-very-secret-dev-key')

# --- Gemini Configuration ---
# genai.configure(api_key="AIzaSyDAy99EYuZqSa6IsRL_YkSLYtGTWGHBiGE")

gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# # --- Database Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'visa_records.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# --- Database Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Look for the 'DATABASE_URL' environment variable (which Render provides)
# 2. If it's not found, fall back to your local SQLite file
database_url = os.environ.get('DATABASE_URL')

if database_url:
    # IMPORTANT: Render provides 'postgres://', but SQLAlchemy requires 'postgresql://'
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'visa_records.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    applications = db.relationship('VisaApplication', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class VisaApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    annual_wage = db.Column(db.Float)
    visa_type = db.Column(db.String(20))
    education = db.Column(db.String(50))
    continent = db.Column(db.String(50))
    region = db.Column(db.String(50))
    has_experience = db.Column(db.String(10))
    no_of_employees = db.Column(db.Integer)
    company_age = db.Column(db.Integer)
    requires_training = db.Column(db.String(10))
    is_full_time = db.Column(db.String(10))
    prev_rejection = db.Column(db.String(10))
    
    status_prediction = db.Column(db.String(20))
    time_prediction = db.Column(db.Integer)
    confidence = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML Model Loading ---
MODELS_PATH = os.path.join(BASE_DIR, 'models')
status_model = joblib.load(os.path.join(MODELS_PATH, 'status_case_model_gb.pkl'))
time_model = joblib.load(os.path.join(MODELS_PATH, 'pt_model.pkl'))

STATUS_FEATURES = ['education_of_employee_High School', 'education_of_employee_Master', 'education_of_employee_Doctorate', 'has_job_experience', 'requires_job_training', 'no_of_employees', 'full_time_position', 'company_age', 'annual_wage', 'continent_Asia', 'continent_Europe', 'continent_North America', 'continent_Oceania', 'continent_South America', 'region_of_employment_South', 'region_of_employment_West']
TIME_FEATURES = ['has_previous_rejection', 'visa_O-1', 'application_month', 'annual_wage', 'visa_H-1B', 'no_of_employees', 'visa_L-1', 'visa_TN']


with app.app_context():
    db.create_all()
# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already registered!')
            return redirect(url_for('signup'))
        
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('assessment'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/assessment')
@login_required
def assessment():
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    user_apps = VisaApplication.query.filter_by(user_id=current_user.id).order_by(VisaApplication.timestamp.desc()).all()
    return render_template('history.html', history=user_apps)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    def get_val(key):
        val = data.get(key, 0)
        try: return float(val) if val != "" else 0.0
        except: return 0.0

    # 1. Input Preparation
    status_df = pd.DataFrame([[get_val(f) for f in STATUS_FEATURES]], columns=STATUS_FEATURES)
    time_df = pd.DataFrame([[get_val(f) for f in TIME_FEATURES]], columns=TIME_FEATURES)

    # 2. Predictions
    # status_pred = status_model.predict(status_df)[0]
    # time_raw_pred = time_model.predict(time_df)[0]
    # status_label = "CERTIFIED" if status_pred == 1 else "DENIED"
    # confidence = round(np.max(status_model.predict_proba(status_df)[0]) * 100, 1) if hasattr(status_model, "predict_proba") else 92.5

    # 2. Predictions (Add float() and int() wrapping)
    status_pred = status_model.predict(status_df)[0]
    time_raw_pred = float(time_model.predict(time_df)[0]) # Convert to float
    status_label = "CERTIFIED" if status_pred == 1 else "DENIED"

    # Convert confidence to a plain Python float
    raw_conf = np.max(status_model.predict_proba(status_df)[0]) if hasattr(status_model, "predict_proba") else 0.925
    confidence = round(float(raw_conf) * 100, 1) # Force float() here

    # 3. Importance logic
    user_status_imp = []
    importances = status_model.feature_importances_
    for i, feat in enumerate(STATUS_FEATURES):
        if get_val(feat) > 0 or feat in ['annual_wage', 'no_of_employees']:
            user_status_imp.append({
                "feature": feat.replace('_', ' ').replace('education of employee', '').replace('continent', 'Origin:').title(),
                "weight": float(importances[i])
            })
    s_total = sum(d['weight'] for d in user_status_imp) or 1
    for d in user_status_imp: d['value'] = d['weight'] / s_total
    user_status_imp = sorted(user_status_imp, key=lambda x: x['value'], reverse=True)[:10]

    user_time_imp = []
    for i, feat in enumerate(TIME_FEATURES):
        if get_val(feat) > 0:
            temp_df = time_df.copy(); temp_df.iloc[0, i] = 0 
            impact = abs(time_raw_pred - time_model.predict(temp_df)[0])
            user_time_imp.append({"feature": feat.replace('_', ' ').title(), "impact": float(impact)})
    t_total = sum(d['impact'] for d in user_time_imp) or 1
    for d in user_time_imp: d['value'] = d['impact'] / t_total
    user_time_imp = sorted(user_time_imp, key=lambda x: x['value'], reverse=True)[:10]

    # 4. AI Insight
    ai_analysis = "Analysis not available."
    try:
        status_chart_summary = ", ".join([f"{d['feature']} ({round(d['value']*100)}% impact)" for d in user_status_imp])

        time_chart_summary = ", ".join([f"{d['feature']} ({round(d['value']*100)}% impact)" for d in user_time_imp])
        
        prompt = f"""
        Act as a Senior US Immigration Analyst. 
        I have a visa prediction with the following data:
        - Result: {status_label}
        - Confidence: {confidence}%
        - Processing Time: {int(time_raw_pred)} days
        - Visa Type: {data.get('visa_category', 'Specified')}

        ML Model Insights:
        1. Factors affecting APPROVAL: {status_chart_summary}
        2. Factors affecting TIME DELAYS: {time_chart_summary}

        Provide a 3-short paragraph 'Strategic Analysis':
        Paragraph 1: Explain why they were {status_label} based on the approval factors.

        Paragraph 2: Analyze the TIME SENSITIVITY. Specifically explain why {user_time_imp[0]['feature']} is impacting their wait time.
        
        Paragraph 3: Give a specific 'Pro-Tip' to optimize their application.
        
        Keep the tone professional and expert. Do not use bullet points.
        """
        response = gemini_model.generate_content(prompt)
        ai_analysis = response.text.strip()
    except: pass

    # --- UPDATED DATABASE LOGIC START ---
    
    # Extract readable strings from One-Hot encoded inputs
    edu_val = "Bachelor"
    for e in ['High School', 'Master', 'Doctorate']:
        if get_val(f'education_of_employee_{e}') == 1: edu_val = e

    cont_val = "Africa"
    for c in ['Asia', 'Europe', 'North America', 'South America', 'Oceania']:
        if get_val(f'continent_{c}') == 1: cont_val = c

    reg_val = "Island"
    for r in ['Midwest', 'Northeast', 'South', 'West']:
        if get_val(f'region_of_employment_{r}') == 1: reg_val = r

    visa_val = "E-2"
    for v in ['H-1B', 'L-1', 'O-1', 'TN']:
        if get_val(f'visa_{v}') == 1: visa_val = v

    # new_record = VisaApplication(
    #     user_id=current_user.id if current_user.is_authenticated else None,
    #     annual_wage=get_val('annual_wage'),
    #     visa_type=visa_val,
    #     education=edu_val,
    #     continent=cont_val,
    #     region=reg_val,
    #     no_of_employees=int(get_val('no_of_employees')),
    #     company_age=int(get_val('company_age')),
    #     has_experience="Yes" if get_val('has_job_experience') == 1 else "No",
    #     requires_training="Yes" if get_val('requires_job_training') == 1 else "No",
    #     is_full_time="Yes" if get_val('full_time_position') == 1 else "No",
    #     prev_rejection="Yes" if get_val('has_previous_rejection') == 1 else "No",
    #     status_prediction=status_label,
    #     time_prediction=int(time_raw_pred),
    #     confidence=confidence
    # )

    new_record = VisaApplication(
        user_id=current_user.id if current_user.is_authenticated else None,
        annual_wage=float(get_val('annual_wage')),
        visa_type=visa_val,
        education=edu_val,
        continent=cont_val,
        region=reg_val,
        no_of_employees=int(get_val('no_of_employees')),
        company_age=int(get_val('company_age')),
        has_experience="Yes" if get_val('has_job_experience') == 1 else "No",
        requires_training="Yes" if get_val('requires_job_training') == 1 else "No",
        is_full_time="Yes" if get_val('full_time_position') == 1 else "No",
        prev_rejection="Yes" if get_val('has_previous_rejection') == 1 else "No",
        status_prediction=status_label,
        time_prediction=int(time_raw_pred), # Ensure int()
        confidence=float(confidence)        # Ensure float()
    )

    db.session.add(new_record)
    db.session.commit()
    
    # --- UPDATED DATABASE LOGIC END ---

    return render_template('result.html', status=status_label, confidence=confidence, days=int(time_raw_pred), 
                           time_range=f"{int(time_raw_pred)-2}-{int(time_raw_pred)+2}", 
                           status_imp=json.dumps(user_status_imp), time_imp=json.dumps(user_time_imp),
                           ai_analysis=ai_analysis, est_date=(datetime.now() + timedelta(days=int(time_raw_pred))).strftime('%b %d, %Y'))

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True, port=5000)