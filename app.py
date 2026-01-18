import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "attribute_lr_best_pipeline.joblib"
CATALOG_PATH = "fully_cleaned_catalog.csv"

# Configure Google Generative AI
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

# --- Global Variables ---
model = None
printer_catalog = None

def load_resources():
    global model, printer_catalog
    # Load scikit-learn model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Classifier model loaded successfully.")
        except Exception as e:
            print(f"Error loading classifier: {e}")
    
    # Load printer catalog
    if os.path.exists(CATALOG_PATH):
        try:
            printer_catalog = pd.read_csv(CATALOG_PATH)
            print(f"Loaded {len(printer_catalog)} products from cleaned catalog.")
        except Exception as e:
            print(f"Error loading catalog: {e}")

# --- Helper Functions ---
def get_gemini_recommendation(user_input, predicted_attributes, chat_history):
    """
    Calls Gemini using the official Google SDK to provide a recommendation.
    Uses the cleaned catalog columns for better context.
    """
    if not api_key:
        return "API Key is missing. Please configure GEMINI_API_KEY in your environment."

    # Filter for only Printers to keep the context focused, but keep other categories available if needed
    printers_only = printer_catalog[printer_catalog['category'] == 'Printer']
    
    # Prepare catalog context using cleaned columns
    # We use 'title', 'price_cleaned', and 'details_cleaned' for the most relevant info
    catalog_context = printers_only[['title', 'price_cleaned', 'details_cleaned']].to_string(index=False)
    
    system_instruction = f"""
    You are an expert Printer Recommendation Assistant. 
    Your goal is to help users find the perfect printer from our catalog.
    
    OUR PRINTER CATALOG (Cleaned Data):
    {catalog_context}
    
    USER ANALYSIS (from our classifier):
    The user's intent suggests these attributes: {predicted_attributes}
    
    INSTRUCTIONS:
    1. Use the predicted attributes and the user's message to filter the catalog.
    2. Recommend 2-3 specific printers that best match their needs.
    3. Use the 'price_cleaned' field for accurate pricing information.
    4. Explain WHY you are recommending them based on their features found in 'details_cleaned'.
    5. If the user's intent is unclear (e.g. missing budget or specific feature needs), ask ONE targeted follow-up question.
    6. Be conversational, professional, and helpful.
    """
    
    try:
        # Initialize the model
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )
        
        # Convert chat history to Google's format
        formatted_history = []
        for msg in chat_history:
            role = 'user' if msg['role'] == 'user' else 'model'
            formatted_history.append({'role': role, 'parts': [msg['content']]})
            
        chat = gemini_model.start_chat(history=formatted_history)
        response = chat.send_message(user_input)
        
        return response.text
    except Exception as e:
        print(f"Gemini SDK Error: {e}")
        return "I'm sorry, I'm having trouble connecting to my recommendation engine. Please check the API key configuration."

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_history = data.get('history', [])
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    # 1. Predict attributes using the scikit-learn model
    predicted_attributes = []
    if model:
        try:
            prediction = model.predict([user_message])
            # In a real scenario, you'd map these to human-readable labels
            predicted_attributes = prediction.tolist()
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # 2. Get recommendation from Gemini
    bot_response = get_gemini_recommendation(user_message, predicted_attributes, chat_history)
    
    return jsonify({
        "response": bot_response,
        "attributes": predicted_attributes
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "classifier_loaded": model is not None,
        "catalog_loaded": printer_catalog is not None,
        "api_key_configured": api_key is not None,
        "catalog_rows": len(printer_catalog) if printer_catalog is not None else 0
    })

# Initialize resources
load_resources()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
