from flask import Flask, request, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('strokemodel.pkl', 'rb') as f:
    model = pickle.load(f)

# Risk Categorization 🎯
risk_levels = [
    (5, "🟢 Very Low Risk", "✅ Maintain a healthy lifestyle!"),
    (10, "🟢 Low Risk", "💪 Keep up good habits like regular exercise!"),
    (15, "🟡 Slight Risk", "🍏 Monitor diet and cholesterol levels."),
    (20, "🟡 Moderate Risk", "🔍 Regular check-ups and a balanced diet recommended."),
    (25, "🟠 Elevated Risk", "⚠️ Manage cholesterol, blood pressure, and lifestyle."),
    (30, "🟠 Concerning Risk", "🩺 Consult a doctor for risk management strategies."),
    (35, "🔴 High Risk", "🚨 Immediate lifestyle changes and medical consultation needed."),
    (40, "🔴 Serious Risk", "🔬 Strictly monitor blood sugar, cholesterol, and blood pressure."),
    (50, "🔴 Critical Risk", "🏥 Consult a doctor and follow strict health guidelines."),
    (100, "🚨 Very High Risk", "⚕️ Urgent medical intervention recommended!"),
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from form ✍️
        name = request.form.get('name', 'User')  # Default name if not provided
        age = int(request.form.get('age', 0))
        sex = request.form.get('sex', 'male')
        bmi = float(request.form.get('bmi', 25.0))  # Default to healthy BMI
        smoking = request.form.get('smoking', '0')
        diabetes = request.form.get('diabetes', 'no')
        hypertension = request.form.get('hypertension', '0')
        atrial_fibrillation = request.form.get('atrial_fibrillation', 'no')
        previous_stroke = request.form.get('previous_stroke', 'no')
        family_history = request.form.get('family_history', 'no')

        # Convert categorical values 🔄
        sex = 1 if sex == 'male' else 0
        smoking = int(smoking)
        diabetes = 1 if diabetes == 'yes' else 0
        hypertension = int(hypertension)
        atrial_fibrillation = 1 if atrial_fibrillation == 'yes' else 0
        previous_stroke = 1 if previous_stroke == 'yes' else 0
        family_history = 1 if family_history == 'yes' else 0

        # Prepare input data 🛠️
        input_data = np.array([
            age, sex, bmi, smoking, diabetes, hypertension, atrial_fibrillation, previous_stroke, family_history
        ]).reshape(1, -1)

        # Scale the input 🔬
        scaled_data = scaler.transform(input_data)

        # Predict stroke probability 📊
        stroke_probability = model.predict_proba(scaled_data)[0][1] * 100  # Probability in %

        # Categorize risk 🎯
        risk_category, advice = next((cat, adv) for max_perc, cat, adv in risk_levels if stroke_probability <= max_perc)

        # Determine risk factors 🔎
        reasons = []
        food_recommendations = []
        
        if age >= 60:
            reasons.append("🔸 **Age is 60 or above**, which increases stroke risk.")
            food_recommendations.append("🥗 **Increase intake of Omega-3 rich foods** like salmon, walnuts, and flaxseeds.")
        
        if bmi < 18.5:
            reasons.append("🔸 **Underweight BMI** can indicate poor nutrition.")
            food_recommendations.append("🍚 **Include nutrient-dense foods** like nuts, avocados, and lean proteins.")
        
        if bmi > 24.9:
            reasons.append("🔸 **Overweight BMI** (above 24.9) increases stroke risk.")
            food_recommendations.append("🥦 **Reduce processed foods and increase fiber intake** (vegetables, whole grains).")

        if smoking:
            reasons.append("🔸 **Smoking** 🌬 damages blood vessels, increasing stroke risk.")
            food_recommendations.append("🍊 **Eat more citrus fruits and green tea** to help detoxify the body.")

        if diabetes:
            reasons.append("🔸 **Diabetes** 🌬 increases stroke risk by damaging blood vessels.")
            food_recommendations.append("🍓 **Increase fiber intake** (berries, legumes, oats) and avoid high-sugar foods.")

        if hypertension:
            reasons.append("🔸 **Hypertension** 💓 (high blood pressure) detected.")
            food_recommendations.append("🥑 **Increase potassium-rich foods** (bananas, spinach, avocados) and reduce salt intake.")

        if atrial_fibrillation:
            reasons.append("🔸 **Atrial fibrillation** ⚡ detected, increasing risk.")
            food_recommendations.append("🍵 **Drink herbal teas** (ginger, green tea) and avoid excessive caffeine.")

        if previous_stroke:
            reasons.append("🔸 **Previous stroke** 🩸 significantly increases the risk of another stroke.")
            food_recommendations.append("🌰 **Include nuts, seeds, and olive oil** to support brain health.")

        if family_history:
            reasons.append("🔸 **Family history of stroke** 🧬 may indicate a genetic predisposition.")
            food_recommendations.append("🥕 **Eat a heart-healthy diet rich in antioxidants** (carrots, leafy greens).")

        # Render results in HTML 📝
        return render_template('result.html', name=name, stroke_probability=round(stroke_probability, 2),
                               risk_category=risk_category, advice=advice, reasons=reasons,
                               food_recommendations=food_recommendations)
    
    except Exception as e:
        return render_template('error.html', error=str(e))
