import unittest
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_prediction(self):
        # Liste de 101 caractéristiques (valeurs fictives pour cet exemple)
        features = [0] * 101  # Remplacez [0] * 101 par les valeurs attendues si nécessaire
        response = self.app.post("/predict", json={"features": features})
        
        # Vérification du statut et des champs dans la réponse
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json)
        self.assertIn("prediction", response.json)

if __name__ == "__main__":
    unittest.main()
