import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = "attribute_lr_best_pipeline.joblib"

# Global variable for the model
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found. Please ensure it is in the same directory.")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No 'text' field provided in request"}), 400
    
    text = data['text']
    
    # Ensure text is in a list if the model expects a batch
    if isinstance(text, str):
        text = [text]
        
    try:
        prediction = model.predict(text)
        # Convert numpy array to list for JSON serialization
        return jsonify({
            "prediction": prediction.tolist(),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    load_model()
    # Use PORT environment variable for cloud deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
else:
    # This block runs when using gunicorn
    load_model()
