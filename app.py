import os
import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
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
def generate_streaming_response(user_input, predicted_attributes, chat_history):
    """
    Generator function that yields chunks of text from Gemini.
    """
    if not api_key:
        yield "data: " + json.dumps({"error": "API Key is missing"}) + "\n\n"
        return

    # Filter for only Printers
    printers_only = printer_catalog[printer_catalog['category'] == 'Printer']
    catalog_context = printers_only[['title', 'price_cleaned', 'details_cleaned', 'image_url']].to_string(index=False)
    
    system_instruction = f"""
    You are a fast, efficient Printer Expert. 
    
    PRIORITY LOGIC:
    1. If the user's request is vague (e.g., missing budget, specific features, or volume needs), DO NOT recommend products yet. Instead, ask ONE short, targeted follow-up question to clarify.
    2. If you have enough info, recommend exactly 2 printers from the catalog below.
    
    CATALOG:
    {catalog_context}
    
    USER INTENT HINTS: {predicted_attributes}
    
    RESPONSE RULES:
    - Be extremely concise to ensure fast response speed.
    - For recommendations, use: ![Product Image](IMAGE_URL)
    - Explain the "Why" in one sentence.
    - If clarifying, be polite but brief.
    """
    
    try:
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 512,
        }

        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction=system_instruction,
            generation_config=generation_config
        )
        
        formatted_history = []
        for msg in chat_history:
            role = 'user' if msg['role'] == 'user' else 'model'
            formatted_history.append({'role': role, 'parts': [msg['content']]})
            
        chat = gemini_model.start_chat(history=formatted_history)
        
        # Use stream=True for real-time response
        response = chat.send_message(user_input, stream=True)
        
        for chunk in response:
            if chunk.text:
                # Send as Server-Sent Events (SSE)
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
    except Exception as e:
        print(f"Gemini Streaming Error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
        
    predicted_attributes = []
    if model:
        try:
            prediction = model.predict([user_message])
            predicted_attributes = prediction.tolist()
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # Return a streaming response
    return Response(
        stream_with_context(generate_streaming_response(user_message, predicted_attributes, chat_history)),
        mimetype='text/event-stream'
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "classifier_loaded": model is not None,
        "catalog_loaded": printer_catalog is not None
    })

load_resources()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
