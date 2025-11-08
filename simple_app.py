"""Simple Flask app for disease prediction - minimal dependencies."""

import json
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load model artifacts
try:
    model = joblib.load("outputs/model.pkl")
    with open("outputs/meta.json", "r") as f:
        meta = json.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    meta = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Disease Prediction Studio</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .upload-area { border: 2px dashed #3498db; padding: 30px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .btn { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #2980b9; }
        .results { margin-top: 20px; padding: 20px; background: #ecf0f1; border-radius: 5px; }
        .error { color: #e74c3c; font-weight: bold; }
        .success { color: #27ae60; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
        th { background: #34495e; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Disease Prediction Studio</h1>
        <p>Upload a CSV file with patient data to get disease predictions.</p>
        
        {% if not model_loaded %}
        <div class="error">❌ Model not loaded. Please train the model first.</div>
        {% else %}
        <div class="success">✅ Model loaded: {{ meta.get('model_type', 'XGBoost') }} for {{ meta.get('n_classes', 'N/A') }} diseases</div>
        {% endif %}
        
        <div class="upload-area">
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required>
                <br><br>
                <button type="submit" class="btn">🔮 Generate Predictions</button>
            </form>
        </div>
        
        {% if error %}
        <div class="results error">{{ error }}</div>
        {% endif %}
        
        {% if predictions %}
        <div class="results">
            <h3>📊 Prediction Results</h3>
            <p><strong>Total Patients:</strong> {{ predictions|length }}</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Patient #</th>
                        <th>Predicted Disease</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {% for i, pred in enumerate(predictions) %}
                    <tr>
                        <td>{{ i + 1 }}</td>
                        <td>{{ pred.disease }}</td>
                        <td>{{ "%.1f%%"|format(pred.confidence * 100) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <div style="margin-top: 30px; text-align: center; color: #7f8c8d;">
            <p>💡 <strong>Required columns:</strong> age, gender, bmi, heart_rate, blood_pressure, cholesterol</p>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(HTML_TEMPLATE, 
                                    model_loaded=model is not None, 
                                    meta=meta)
    
    # Handle file upload
    if not model:
        return render_template_string(HTML_TEMPLATE, 
                                    model_loaded=False, 
                                    meta=meta,
                                    error="Model not loaded")
    
    try:
        file = request.files["file"]
        if not file or not file.filename.endswith('.csv'):
            raise ValueError("Please upload a CSV file")
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Basic validation
        required_cols = ["age", "gender", "bmi", "heart_rate", "blood_pressure", "cholesterol"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
        
        # Make predictions
        try:
            probabilities = model.predict_proba(df)
            predictions = []
            
            if probabilities.shape[1] == 2:  # Binary classification
                for i, prob in enumerate(probabilities):
                    pred_class = 1 if prob[1] > 0.5 else 0
                    confidence = max(prob)
                    disease = "Disease Detected" if pred_class == 1 else "Healthy"
                    predictions.append({"disease": disease, "confidence": confidence})
            else:  # Multi-class
                pred_classes = model.predict(df)
                for i, (pred_class, prob_row) in enumerate(zip(pred_classes, probabilities)):
                    confidence = max(prob_row)
                    disease = f"Disease Class {pred_class}"
                    predictions.append({"disease": disease, "confidence": confidence})
        
        except Exception as pred_error:
            raise ValueError(f"Prediction failed: {str(pred_error)}")
        
        return render_template_string(HTML_TEMPLATE, 
                                    model_loaded=True, 
                                    meta=meta,
                                    predictions=predictions)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, 
                                    model_loaded=True, 
                                    meta=meta,
                                    error=str(e))

if __name__ == "__main__":
    print("🚀 Starting Disease Prediction Studio...")
    print("📍 Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
