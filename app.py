import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# The model file name must match the one you upload
MODEL_PATH = "attribute_lr_best_pipeline.joblib"

# Global variable for the model
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Load the model using joblib
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found. Deployment will fail if not present.")

@app.route('/', methods=['GET'])
def index():
    # Serve the HTML UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Check if request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No 'text' field provided in request"}), 400
    
    text = data['text']
    
    # Ensure text is in a list as the scikit-learn pipeline expects an iterable
    if isinstance(text, str):
        text = [text]
        
    try:
        # Perform prediction
        prediction = model.predict(text)
        
        # Convert numpy array to list for JSON serialization
        return jsonify({
            "prediction": prediction.tolist(),
            "status": "success"
        })
    except Exception as e:
        # Log the error and return a 500 status
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "environment": os.environ.get("RENDER", "local")
    })

# Load the model when the application starts
load_model()

if __name__ == '__main__':
    # Use PORT environment variable for cloud deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
