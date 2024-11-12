import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftMetric
from evidently.dashboard import Dashboard
import webbrowser

# Étape 1 : Charger les données
train_data = pd.read_csv(r"C:\Users\SOUKA\Desktop\Projet7_IMS\Projet+Mise+en+prod+-+home-credit-default-risk (1)\application_train.csv")
test_data = pd.read_csv(r"C:\Users\SOUKA\Desktop\Projet7_IMS\Projet+Mise+en+prod+-+home-credit-default-risk (1)\application_test.csv")

# Étape 2 : Configurer les colonnes pour l'analyse
column_mapping = ColumnMapping(
    target="TARGET",  # Remplacez par le nom de votre colonne cible dans application_train
    features=["feature1", "feature2", "feature3", "feature4", "feature5",  # Remplacez par vos caractéristiques importantes
              "feature6", "feature7", "feature8", "feature9", "feature10"]
)

# Étape 3 : Créer le rapport d'analyse
report = Report(metrics=[DataDriftMetric()])

# Ajouter les données d'entraînement et de test
report.run(train_data, test_data, column_mapping)

# Étape 4 : Sauvegarder le rapport dans un fichier HTML
report.save_html("data_drift_report.html")

# Étape 5 : Ouvrir et analyser le rapport
webbrowser.open("data_drift_report.html")
