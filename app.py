import os
import json
import joblib
import pandas as pd
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "attribute_lr_best_pipeline.joblib")
CATALOG_PATH = os.path.join(BASE_DIR, "fully_cleaned_catalog.csv")

# Hugging Face Configuration
HF_API_URL = os.environ.get("HF_API_URL") # e.g., https://api-inference.huggingface.co/models/your-username/your-model
HF_TOKEN = os.environ.get("HF_TOKEN")     # Your HF Access Token

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 # 10MB limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# --- Global Resources ---
text_model = None
text_catalog = None

def load_resources():
    global text_model, text_catalog
    
    # Load SM2 Text Resources
    if os.path.exists(MODEL_PATH):
        text_model = joblib.load(MODEL_PATH)
        print("SM2 Classifier loaded.")
    if os.path.exists(CATALOG_PATH):
        text_catalog = pd.read_csv(CATALOG_PATH)
        print("Product catalog loaded.")

# --- SM2 Text Chat Logic ---
def generate_streaming_response(user_input, predicted_attributes, chat_history):
    if not api_key:
        yield f"data: {json.dumps({'error': 'API Key missing'})}\n\n"
        return

    # Filter for printers to provide context to Gemini
    if text_catalog is not None:
        printers_only = text_catalog[text_catalog['category'] == 'Printer']
        catalog_context = printers_only[['title', 'price_cleaned', 'details_cleaned', 'image_url']].head(15).to_string(index=False)
    else:
        catalog_context = "Catalog not available."
    
    system_instruction = f"""
    You are a fast, efficient Printer Expert. 
    PRIORITY: If vague, ask ONE short follow-up. If clear, recommend 2 printers.
    CATALOG: {catalog_context}
    USER INTENT HINTS: {predicted_attributes}
    RULES: Concise. Use ![Product Image](IMAGE_URL). Explain 'Why' in one sentence.
    """
    
    try:
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system_instruction)
        formatted_history = [{'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']]} for m in chat_history]
        chat = gemini_model.start_chat(history=formatted_history)
        response = chat.send_message(user_input, stream=True)
        for chunk in response:
            if chunk.text:
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# --- SM4 Visual Search Logic (Remote via Hugging Face) ---
def search_visual_catalog_remote(img_path):
    if not HF_API_URL:
        return {"ok": False, "error": "Hugging Face API URL not configured."}
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    try:
        with open(img_path, "rb") as f:
            # Note: This assumes the HF Space expects a file upload and returns JSON results
            # You might need to adjust this based on how your teammate's Space is set up
            response = requests.post(HF_API_URL, headers=headers, files={"file": f}, timeout=30)
            
        if response.status_code == 200:
            return {"ok": True, "data": response.json()}
        else:
            return {"ok": False, "error": f"HF API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_history = data.get('history', [])
    
    predicted_attributes = []
    if text_model:
        try:
            prediction = text_model.predict([user_message])
            predicted_attributes = prediction.tolist()
        except:
            predicted_attributes = ["Error in classification"]
    
    return Response(stream_with_context(generate_streaming_response(user_message, predicted_attributes, chat_history)), mimetype='text/event-stream')

@app.route("/api/search", methods=["POST"])
def api_search():
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    safe_name = secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}")
    file.save(saved_path)

    # Call the remote Hugging Face API
    hf_result = search_visual_catalog_remote(saved_path)
    
    if not hf_result["ok"]:
        return jsonify({"ok": False, "error": hf_result["error"]}), 500

    # Return the results from Hugging Face to the frontend
    # This assumes the HF Space returns a structure similar to what the UI expects
    return jsonify({
        "ok": True,
        "query_image_url": f"/static/uploads/{os.path.basename(saved_path)}",
        "hf_data": hf_result["data"] # Pass the raw HF data to the frontend
    })

load_resources()

if __name__ == "__main__":
    app.run(debug=True, port=5000)