import pandas as pd
import streamlit as st
import requests

# URL de l'API déployée
API_URL = "https://p7-api-d-ploiement-1.onrender.com/"

# Fonction pour obtenir la prédiction depuis l'API
def get_prediction(features):
    # S'assurer que le nombre de caractéristiques est 101
    if len(features) < 101:
        features = features + [0] * (101 - len(features))  # Compléter avec des zéros
    features = features[:101]  # S'assurer que nous avons exactement 101 caractéristiques

    response = requests.post(API_URL + "predict", json={"features": features})
    if response.status_code == 200:
        result = response.json()
        return result.get("prediction"), result.get("probability")
    else:
        st.error("Erreur dans la réponse de l'API.")
        return None, None

# Chargement des données des clients
clients_df = pd.read_csv(r"C:\Users\SOUKA\Desktop\Projet7_IMS\df_top_features_cleaned.xls")

# Extraction des caractéristiques importantes
important_features = clients_df[['INS_D365DPD_DIFF_MAX', 'INS_D365INS_IS_DPD_UNDER_120_MEAN', 
                                  'BURO_ENDDATE_DIF_MAX', 'EXT_SOURCES_MAX', 'CREDIT_TO_GOODS_RATIO', 
                                  'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'EXT_SOURCES_MIN', 
                                  'POS_CNT_INSTALMENT_MIN', 'PREV_DAYS_LAST_DUE_DIFF_MEAN', 
                                  'PREV_APP_CREDIT_PERC_VAR']]

# Titre de l'application
st.title("Simulation de Scoring Client")

# Sélection du client
client_id = st.selectbox("Sélectionnez un client", options=clients_df.index)

# Caractéristiques du client sélectionné
features = important_features.iloc[client_id].tolist()

# Bouton pour lancer la prédiction
if st.button("Calculer le Score"):
    prediction, probability = get_prediction(features)
    
    if prediction is not None:
        # Affichage des résultats
        st.write(f"Prédiction : {'Crédit Accordé' if prediction == 1 else 'Crédit Refusé'}")
        st.write(f"Probabilité de défaut : {probability:.2f}")
        
        # Définition du seuil
        seuil = 0.24  # Ajustez ce seuil selon votre stratégie de scoring
        decision = "Crédit Accordé" if probability >= seuil else "Crédit Refusé"
        st.write(f"Décision basée sur le seuil : {decision}")
