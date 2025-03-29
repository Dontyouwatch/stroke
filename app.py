from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and columns
model = joblib.load('stroke_model.joblib')
scaler = joblib.load('scaler.joblib')
input_columns = joblib.load('columns.joblib')

def prepare_input(data):
    age = float(data['Age'])
    bmi = float(data['BMI'])
    cholesterol = float(data['Cholesterol'])
    hypertension = int(data['Hypertension_Category'])
    atrial_fibrilation = int(data['Atrial_Fibrilation'])
    diabetes = int(data['Diabetes'])
    smoking = int(data['Smoking'])
    previous_stroke = int(data['Previous_Stroke'])

    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 24.9 else 2 if bmi < 29.9 else 3
    cholesterol_category = 0 if cholesterol < 180 else 1 if cholesterol < 240 else 2
    age_group = 0 if age < 40 else 1 if age < 60 else 2
    bmi_hypertension = bmi * hypertension
    cholesterol_afib = cholesterol * atrial_fibrilation

    user_input = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Cholesterol": cholesterol,
        "Hypertension_Category": hypertension,
        "Atrial_Fibrilation": atrial_fibrilation,
        "Diabetes": diabetes,
        "Smoking": smoking,
        "Previous_Stroke": previous_stroke,
        "BMI_Hypertension": bmi_hypertension,
        "Cholesterol_AFib": cholesterol_afib,
        "BMI_Category_1": 1 if bmi_category == 1 else 0,
        "BMI_Category_2": 1 if bmi_category == 2 else 0,
        "BMI_Category_3": 1 if bmi_category == 3 else 0,
        "Cholesterol_Category_1": 1 if cholesterol_category == 1 else 0,
        "Cholesterol_Category_2": 1 if cholesterol_category == 2 else 0,
        "Age_Group_1": 1 if age_group == 1 else 0,
        "Age_Group_2": 1 if age_group == 2 else 0
    }])

    # Align input features with training data
    missing_cols = set(input_columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0

    user_input = user_input[input_columns]
    numeric_features = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_AFib"]
    user_input[numeric_features] = scaler.transform(user_input[numeric_features])
    return user_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = prepare_input(data)
        prediction = model.predict_proba(input_data)[:, 1][0]
        result = "Yes" if prediction > 0.5 else "No"
        return jsonify({'stroke_risk': result, 'probability': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
