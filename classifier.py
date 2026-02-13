import joblib
import os
from config import Config

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(Config.MODEL_PATH):
            try:
                self.model = joblib.load(Config.MODEL_PATH)
                print("SM2 Classifier loaded successfully.")
            except Exception as e:
                print(f"Error loading SM2 model: {e}")
        else:
            print(f"Model file not found at {Config.MODEL_PATH}")

    def predict(self, text):
        if self.model:
            try:
                # Assuming the model returns a list of binary attributes
                prediction = self.model.predict([text])
                return prediction.tolist()[0]
            except Exception as e:
                print(f"Prediction error: {e}")
                return []
        return []

    def get_attribute_names(self):
        # Based on user's previous description
        return ["HomeUse", "Wireless", "Duplex", "BudgetConcern"]