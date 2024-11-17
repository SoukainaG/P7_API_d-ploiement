# Importer les bibliothèques nécessaires
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(page_title="Tableau de Bord de Scoring Client", page_icon=":bar_chart:")

# URL brute de votre fichier CSV sur GitHub
csv_url = 'https://raw.githubusercontent.com/SoukainaG/P7_API_d-ploiement/main/df_final_cleaned_S.csv'

# Charger les données directement depuis GitHub
try:
    clients_df = pd.read_csv(csv_url)
    st.success("Les données ont été chargées avec succès depuis GitHub.")
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

# Affichage d'un aperçu des données
st.write("### Aperçu des données")
st.write(clients_df.head())

# URL de l'API déployée
API_URL = "https://p7-api-d-ploiement-6.onrender.com/"

# Fonction pour obtenir la prédiction depuis l'API
def get_prediction(features):
    if len(features) < 73:
        features = features + [0] * (73 - len(features))  # Compléter avec des zéros si nécessaire
    response = requests.post(API_URL + "predict", json={"features": features})
    if response.status_code == 200:
        result = response.json()
        return result.get("prediction"), result.get("probability")
    else:
        st.error("Erreur dans la réponse de l'API.")
        return None, None

# Extraire les caractéristiques importantes
important_features = clients_df[['EXT_SOURCES_MAX', 'HOUR_APPR_PROCESS_START', 'EXT_SOURCES_MIN', 
                                  'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'CREDIT_TO_GOODS_RATIO', 
                                  'INS_D365INS_IS_DPD_UNDER_120_MEAN', 'POS_CNT_INSTALMENT_MIN', 
                                  'PREV_HOUR_APPR_PROCESS_START_MAX', 'INSTAL_INS_IS_DPD_UNDER_120_MEAN', 
                                  'PREV_DAYS_TERMINATION_MAX', 'PREV_APP_CREDIT_PERC_MAX', 
                                  'PREV_DAYS_LAST_DUE_DIFF_MEAN', 'INSTAL_LATE_PAYMENT_MEAN', 
                                  'INS_D365DPD_DIFF_MAX', 'INCOME_TO_EMPLOYED_RATIO', 
                                  'EXT_SOURCE_2', 'REGION_POPULATION_RELATIVE', 
                                  'PREV_SIMPLE_INTERESTS_MEAN', 'PAYMENT_RATE', 'DAYS_EMPLOYED']]

# Interface utilisateur : Sélection du client
st.sidebar.header("Sélection du Client")
client_id = st.sidebar.selectbox("ID du Client :", options=clients_df.index)

# Récupérer les caractéristiques pour le client sélectionné
client_features = important_features.iloc[client_id].tolist()

# Obtenir la prédiction pour le client sélectionné
if 'prediction' not in st.session_state or st.session_state.get("client_id") != client_id:
    prediction, probability = get_prediction(client_features)
    st.session_state["prediction"] = prediction
    st.session_state["probability"] = probability
    st.session_state["client_id"] = client_id

# Afficher les résultats de la prédiction
if st.session_state.get("prediction") is not None:
    seuil = 0.24
    decision = "Crédit Accordé" if st.session_state["probability"] >= seuil else "Crédit Refusé"

    st.write("## Résultats pour le Client Sélectionné")
    st.write(f"**Prédiction :** {'Crédit Refusé' if st.session_state['prediction'] == 1 else 'Crédit Accordé'}")
    st.write(f"**Probabilité de défaut :** {st.session_state['probability']:.2f}")
    st.write(f"**Décision :** {decision}")

    # Jauge pour afficher le score
    st.write("### Score de Probabilité")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state["probability"],
        title={'text': "Score de Probabilité"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, seuil], 'color': "red"},
                         {'range': [seuil, 1], 'color': "green"}]}
    ))
    st.plotly_chart(fig_gauge)

# Analyse des caractéristiques
st.write("## Analyse des Caractéristiques")

# Sélection d'une caractéristique pour l'analyse univariée
feature1 = st.selectbox("Sélectionnez une caractéristique :", important_features.columns)
st.write(f"### Distribution de {feature1}")
fig_feature1 = px.histogram(clients_df, x=feature1, color="TARGET", marginal="box", nbins=30)
fig_feature1.add_vline(x=client_features[important_features.columns.get_loc(feature1)], 
                        line_dash="dash", line_color="red")
st.plotly_chart(fig_feature1)

# Analyse bi-variée
st.write("## Analyse Bi-Variée")
feature2 = st.selectbox("Sélectionnez une deuxième caractéristique :", important_features.columns)
fig_bivar = px.scatter(clients_df, x=feature1, y=feature2, color="TARGET",
                       title=f"Relation entre {feature1} et {feature2}")
st.plotly_chart(fig_bivar)

# Importance globale des caractéristiques
st.write("## Importance Globale des Caractéristiques")
global_importance = clients_df[important_features.columns].mean().sort_values(ascending=False)
fig_global = px.bar(global_importance, x=global_importance.values, y=global_importance.index, orientation='h')
st.plotly_chart(fig_global)
