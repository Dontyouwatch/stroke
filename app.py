
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('strokemodel.pkl', 'rb') as f:
    model = pickle.load(f)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        name = request.form.get('name')
        age = int(request.form.get('age'))
        sex = request.form.get('sex')
        bmi = float(request.form.get('bmi'))
        smoking = request.form.get('smoking')
        diabetes = request.form.get('diabetes')
        hypertension = request.form.get('hypertension')
        atrial_fibrillation = request.form.get('atrial_fibrillation')
        previous_stroke = request.form.get('previous_stroke')
        family_history = request.form.get('family_history')

        # Convert string values to numerical based on your model's expectations
        sex = 1 if sex == 'male' else 0  # Assuming 1 for male, 0 for female
        smoking = int(smoking) # 1 for smoker, 0 for non-smoker
        diabetes = 1 if diabetes == 'yes' else 0
        hypertension = int(hypertension)  # Already numerical based on the option values
        atrial_fibrillation = 1 if atrial_fibrillation == 'yes' else 0
        previous_stroke = 1 if previous_stroke == 'yes' else 0
        family_history = 1 if family_history == 'yes' else 0

        # Prepare input for the model
        input_data = np.array([
            age, sex, bmi, smoking, diabetes, hypertension, atrial_fibrillation, previous_stroke, family_history
        ]).reshape(1, -1)

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        stroke_probability = model.predict_proba(scaled_data)[0][1] * 100  # Probability in %

        # Risk categorization
        risk_category, advice = next((cat, adv) for max_perc, cat, adv in risk_levels if stroke_probability <= max_perc)

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

        # Return result as JSON
        return jsonify({
            "name": name,
            "stroke_percentage": round(stroke_probability, 2),
            "risk_category": risk_category,
            "advice": advice,
            "reasons": reasons,
            "recommended_foods": recommended_foods
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
