# Import des librairies
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Chargement du modèle avec joblib
model = joblib.load("lightgbm_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération des données JSON
        data = request.get_json(force=True)
        
        # Vérifier si 'features' est présent dans les données
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' key in request data"}), 400
        
        # Récupérer les caractéristiques
        features = data['features']
        
        # S'assurer que le nombre de caractéristiques est correct
        if len(features) != 101:  # Supposons que votre modèle nécessite 101 caractéristiques
            # Remplir les valeurs manquantes avec des zéros
            features = features + [0] * (101 - len(features))  # Compléter jusqu'à 101
            features = features[:101]  # S'assurer que nous avons exactement 101 caractéristiques

        # Faire la prédiction
        prediction = model.predict([features])
        probability = model.predict_proba([features])[0, 1]
        
        # Conversion en types JSON-serialisables
        return jsonify({
            "prediction": int(prediction[0]), 
            "probability": float(probability)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
