from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("flood_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "annual_rainfall", "cloud_visibility", "monsoon_rainfall",
    "pre_monsoon_rainfall", "post_monsoon_rainfall", "winter_rainfall"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(request.form[f]) for f in FEATURES]
        prediction = model.predict([data])[0]
        proba = model.predict_proba([data])[0][1] if hasattr(model, "predict_proba") else None
        result = {
            "prediction": int(prediction),
            "label": "High Flood Risk" if prediction == 1 else "Low Flood Risk",
            "probability": f"{proba * 100:.1f}%" if proba is not None else "N/A"
        }
        return render_template("index.html", result=result)
    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json()
        data = [float(body[f]) for f in FEATURES]
        prediction = model.predict([data])[0]
        proba = model.predict_proba([data])[0][1] if hasattr(model, "predict_proba") else None
        return jsonify({
            "prediction": int(prediction),
            "label": "High Flood Risk" if prediction == 1 else "Low Flood Risk",
            "probability": round(proba * 100, 2) if proba is not None else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
