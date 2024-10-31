from flask import Flask, request, jsonify
import joblib

# Chargement du modèle avec joblib
model = joblib.load("lightgbm_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    probability = model.predict_proba([data['features']])[0, 1]
    
    # Conversion en types JSON-serialisables
    return jsonify({
        "prediction": int(prediction[0]), 
        "probability": float(probability)
    })

@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de prédiction ! Utilisez /predict pour obtenir des prédictions.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
