from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the app


# Load the scaler and model
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('strokemodel.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    #  Handle this error.  The app should not run without the model and scaler.
    #  A better approach would be to raise an exception, or sys.exit()
    model = None
    scaler = None



# Risk Categorization
risk_levels = [
    (5, "🟢 Very Low Risk", "Maintain a healthy lifestyle."),
    (10, "🟢 Low Risk", "Keep up good habits like regular exercise."),
    (15, "🟡 Slight Risk", "Monitor diet and cholesterol."),
    (20, "🟡 Moderate Risk", "Regular check-ups and a balanced diet recommended."),
    (25, "🟠 Elevated Risk", "Manage cholesterol, blood pressure, and lifestyle."),
    (30, "🟠 Concerning Risk", "Consult a doctor for risk management strategies."),
    (35, "🔴 High Risk", "Immediate lifestyle changes and medical consultation needed."),
    (40, "🔴 Serious Risk", "Strictly monitor blood sugar, cholesterol, and blood pressure."),
    (50, "🔴 Critical Risk", "Consult a doctor and follow strict health guidelines."),
    (100, "🚨 Very High Risk", "Urgent medical intervention recommended."),
]

# Gender-Specific Food Recommendations
foods = {
    "low": {
        "male": ["Salmon 🐟", "Avocado 🥑", "Almonds 🌰", "Leafy greens 🥬", "Oatmeal 🥣"],
        "female": ["Berries 🍓", "Greek yogurt 🥛", "Dark chocolate 🍫", "Walnuts 🌰", "Spinach 🥬"]
    },
    "moderate": {
        "male": ["Lean chicken 🍗", "Brown rice 🍚", "Chia seeds 🌱", "Broccoli 🥦", "Olive oil 🫒"],
        "female": ["Quinoa 🍚", "Lentils 🥣", "Salmon 🐟", "Flaxseeds 🌱", "Carrots 🥕"]
    },
    "high": {
        "male": ["Tofu 🍢", "Beets 🍠", "Garlic 🧄", "Turmeric 🌿", "Dark chocolate 🍫"],
        "female": ["Sweet potatoes 🍠", "Soy milk 🥛", "Chia pudding 🍮", "Pumpkin seeds 🌰", "Tomatoes 🍅"]
    },
    "critical": {
        "male": ["Boiled vegetables 🥕", "Steamed fish 🐟", "Green tea 🍵", "Whole wheat 🥖", "Berries 🍓"],
        "female": ["Almond milk 🥛", "Cottage cheese 🧀", "Kale 🥬", "Apple cider vinegar 🍏", "Oats 🥣"]
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Model or scaler not loaded.  The application cannot run.", 500
    try:
        # Get data from form
        name = request.form.get('name')
        age = int(request.form.get('age'))
        sex = request.form.get('sex')
        bmi = float(request.form.get('bmi'))
        smoking = request.form.get('smoking')
        diabetes = request.form.get('diabetes')
        hypertension = int(request.form.get('hypertension'))
        atrial_fibrillation = request.form.get('atrial-fibrillation')
        previous_stroke = request.form.get('previous-stroke')
        family_history = request.form.get('family-history')

        # Convert string values to numerical
        sex = 1 if sex == 'male' else 0
        smoking = int(smoking)  # 1 for smoker, 0 for non-smoker
        diabetes = 1 if diabetes == 'yes' else 0
        atrial_fibrillation = 1 if atrial_fibrillation == 'yes' else 0
        previous_stroke = 1 if previous_stroke == 'yes' else 0
        family_history = 1 if family_history == 'yes' else 0

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'BMI': bmi,
            'Smoking': smoking,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'AFib': atrial_fibrillation,
            'Previous_Stroke': previous_stroke,
            'Family_History': family_history
        }])

        # Feature Engineering (same as in training - IMPORTANT)
        input_data["BMI_Category"] = pd.cut(input_data["BMI"], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=[0, 1, 2, 3]).astype(int)
        input_data["Cholesterol_Category"] = pd.cut(pd.Series([200]), bins=[0, 180, 240, np.inf], labels=[0, 1, 2]).astype(int)  # Use a dummy value
        input_data["Age_Group"] = pd.cut(input_data["Age"], bins=[0, 40, 60, np.inf], labels=[0, 1, 2]).astype(int)
        input_data["BMI_Hypertension"] = input_data["BMI"] * input_data["Hypertension"]
        input_data["Cholesterol_AFib"] = 200 * input_data["AFib"]  # Use dummy

        # Convert categorical features to numerical using get_dummies
        input_data = pd.get_dummies(input_data, columns=["BMI_Category", "Cholesterol_Category", "Age_Group"], drop_first=True)

        # Scale the input data
        numeric_features = ["Age", "BMI", "BMI_Hypertension", "Cholesterol_AFib"]
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

        # Align the input data with the columns used during training.   CRUCIAL
        expected_cols = ['Age', 'Sex', 'BMI', 'Smoking', 'Diabetes', 'Hypertension', 'AFib', 'Previous_Stroke',
                         'Family_History', 'BMI_Hypertension', 'Cholesterol_AFib', 'BMI_Category_1', 'BMI_Category_2',
                         'BMI_Category_3', 'Cholesterol_Category_1', 'Cholesterol_Category_2', 'Age_Group_1', 'Age_Group_2']
        missing_cols = set(expected_cols) - set(input_data.columns)
        for c in missing_cols:
            input_data[c] = 0
        input_data = input_data[expected_cols]

        # Make prediction
        stroke_probability = model.predict_proba(input_data)[0][1] * 100  # Probability in %

        # Risk categorization
        risk_category, advice = next(
            (cat, adv) for max_perc, cat, adv in risk_levels if stroke_probability <= max_perc)

        # Explanation based on user input
        reasons = []
        if age >= 60:
            reasons.append("🔸 **Age is 60 or above**, which increases stroke risk.")
        if bmi < 18.5 or bmi > 24.9:
            reasons.append("🔸 **Unhealthy BMI** (underweight or overweight).")
        if smoking:
            reasons.append("🔸 **Smoking** damages blood vessels, increasing stroke risk.")
        if diabetes:
            reasons.append("🔸 **Diabetes** increases the risk of stroke by damaging blood vessels.")
        if hypertension:
            reasons.append("🔸 **Hypertension** (high blood pressure) detected.")
        if atrial_fibrillation:
            reasons.append("🔸 **Atrial fibrillation** detected, increasing risk.")
        if previous_stroke:
            reasons.append("🔸 **Previous stroke** significantly increases the risk of another stroke.")
        if family_history:
            reasons.append("🔸 **Family history of stroke** may indicate a genetic predisposition.")

        # Food recommendations
        diet_key = "low" if stroke_probability <= 10 else \
            "moderate" if stroke_probability <= 20 else \
            "high" if stroke_probability <= 35 else "critical"
        recommended_foods = foods[diet_key]["male" if sex == 1 else "female"]

        # Render the result.html template with the prediction data
        return render_template('result.html',
                               name=name,
                               stroke_probability=round(stroke_probability, 2),
                               risk_category=risk_category,
                               advice=advice,
                               reasons=reasons,
                               food_recommendation=recommended_foods)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return JSON error with 500 status code
