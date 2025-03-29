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
        if age >= 60:
            reasons.append("🔸 **Age is 60 or above**, which increases stroke risk.")
        if bmi < 18.5 or bmi > 24.9:
            reasons.append("🔸 **Unhealthy BMI** (underweight or overweight).")
        if smoking:
            reasons.append("🔸 **Smoking** 🚬 damages blood vessels, increasing stroke risk.")
        if diabetes:
            reasons.append("🔸 **Diabetes** 🍬 increases stroke risk by damaging blood vessels.")
        if hypertension:
            reasons.append("🔸 **Hypertension** 💓 (high blood pressure) detected.")
        if atrial_fibrillation:
            reasons.append("🔸 **Atrial fibrillation** ⚡ detected, increasing risk.")
        if previous_stroke:
            reasons.append("🔸 **Previous stroke** 🩸 significantly increases the risk of another stroke.")
        if family_history:
            reasons.append("🔸 **Family history of stroke** 🧬 may indicate a genetic predisposition.")
        
        # Physical Activity Recommendations 🏃‍♂️
        if stroke_probability < 15:
            activity_suggestion = "🏋️ **Regular Exercise**: Continue engaging in at least 30 minutes of moderate exercise daily."
        elif stroke_probability < 30:
            activity_suggestion = "🚶 **Increase Physical Activity**: Brisk walking, jogging, or swimming can help maintain good heart health."
        else:
            activity_suggestion = "⚠️ **Consult a Doctor**: Before engaging in any intense physical activity, consult your physician."

        # Dietary Recommendations 🍎
        if stroke_probability < 15:
            diet_suggestion = "🥗 **Balanced Diet**: Continue eating fiber-rich foods like fruits, vegetables, and whole grains."
            food_suggestion = "🍏 Apples, 🥦 Broccoli, 🥜 Almonds, 🥒 Cucumbers, 🍓 Berries"
        elif stroke_probability < 30:
            diet_suggestion = "🍣 **Heart-Healthy Foods**: Include omega-3 fatty acids from fish, nuts, and seeds."
            food_suggestion = "🐟 Salmon, 🥑 Avocado, 🌰 Walnuts, 🍊 Oranges, 🥬 Spinach"
        elif stroke_probability < 50:
            diet_suggestion = "🛑 **Reduce Sodium & Sugars**: Avoid processed foods, high salt intake, and sugary beverages."
            food_suggestion = "🍗 Lean chicken, 🍚 Brown rice, 🫑 Bell peppers, 🥕 Carrots, 🥦 Kale"
        else:
            diet_suggestion = "🚨 **Strict Diet Control**: Avoid all processed foods, high-fat dairy, and excessive sodium."
            food_suggestion = "🍵 Green tea, 🥒 Leafy greens, 🫘 Lentils, 🍠 Sweet potatoes, 🍊 Citrus fruits"

        # Render results in HTML 📄
        return render_template('result.html', name=name, stroke_probability=round(stroke_probability, 2),
                               risk_category=risk_category, advice=advice, reasons=reasons,
                               activity_suggestion=activity_suggestion, diet_suggestion=diet_suggestion,
                               food_suggestion=food_suggestion)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
