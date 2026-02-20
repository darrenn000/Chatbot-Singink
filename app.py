import os
import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import google.generativeai as genai
from dotenv import load_dotenv
import re
from gradio_client import Client, handle_file
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "attribute_lr_best_pipeline.joblib"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "fully_cleaned_catalog.csv")
MODEL_PATH = os.path.join(BASE_DIR, "attribute_lr_best_pipeline.joblib")

# SM4 Space configuration
SM4_SPACE_ID = os.environ.get("SM4_SPACE_ID", "lirou-0/sm4_visual_similarity")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only if space is private

sm4_client = None

def get_sm4_client():
    global sm4_client
    if sm4_client is None:
        if HF_TOKEN:
            sm4_client = Client(SM4_SPACE_ID, hf_token=HF_TOKEN)
        else:
            sm4_client = Client(SM4_SPACE_ID)
    return sm4_client

# --- SM4 Upload Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_MB = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def select_candidates(user_message: str, catalog_df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """
    Cheap Top-K retrieval to avoid sending the full catalog to Gemini.
    """
    df = catalog_df[catalog_df["category"] == "Printer"].copy()
    msg = _norm(user_message)

    # Heuristic boosts (optional but useful)
    if any(w in msg for w in ["wireless", "wifi", "wi-fi"]):
        df["_wireless"] = df["details_cleaned"].fillna("").str.lower().str.contains("wifi|wi-fi|wireless")
        df = df.sort_values("_wireless", ascending=False)

    if any(w in msg for w in ["duplex", "double-sided", "double sided"]):
        df["_duplex"] = df["details_cleaned"].fillna("").str.lower().str.contains("duplex|double[- ]sided")
        df = df.sort_values("_duplex", ascending=False)

    # Relevance scoring
    terms = [t for t in re.findall(r"[a-z0-9]+", msg) if len(t) >= 3][:10]
    text = (df["title"].fillna("") + " " + df["details_cleaned"].fillna("")).str.lower()

    df["_score"] = 0
    for t in terms:
        df["_score"] += text.str.count(re.escape(t))

    return df.sort_values("_score", ascending=False).head(k)

def pack_products(df: pd.DataFrame) -> list:
    """
    Hard-trim product text to control tokens.
    """
    packed = []
    for _, r in df.iterrows():
        packed.append({
            "title": str(r.get("title", ""))[:120],
            "price": r.get("price_cleaned", None),
            "details": str(r.get("details_cleaned", ""))[:260],   # IMPORTANT: truncate
            "image_url": r.get("image_url", ""),
        })
    return packed


def generate_streaming_response(user_input, predicted_attributes, chat_history):
    """
    Generator function that yields chunks of text from Gemini.
    Token-optimized: retrieve Top-K candidates, send only those.
    """
    if not api_key:
        yield "data: " + json.dumps({"error": "API Key is missing"}) + "\n\n"
        return

    # Safety: if catalog not loaded
    if printer_catalog is None or printer_catalog.empty:
        yield "data: " + json.dumps({"error": "Catalog not loaded"}) + "\n\n"
        return

    # 1) Trim history (do NOT resend everything forever)
    chat_history = (chat_history or [])[-4:]

    # 2) Retrieve Top-K candidates instead of full catalog
    candidates_df = select_candidates(user_input, printer_catalog, k=12)
    candidates = pack_products(candidates_df)

    # 3) Tiny system instruction (no catalog inside)
    system_instruction = (
        "You are a Printer Recommendation Formatter.\n"
        "Given user message + predicted attributes + candidate list:\n"
        "If info missing, ask exactly ONE short follow-up question.\n"
        "Otherwise recommend exactly 2 products from candidates.\n"
        "Output format:\n"
        "- If clarify: one short question only.\n"
        "- If recommend: for each product include: ![Product Image](IMAGE_URL) then 1-sentence why.\n"
        "Be concise. Do not mention candidates list."
    )

    # 4) Send compact JSON payload to Gemini
    payload = {
        "user_message": user_input[:800],  # hard cap user text
        "predicted_attributes": predicted_attributes,
        "candidates": candidates
    }

    try:
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 250,  # reduced from 512
        }

        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction=system_instruction,
            generation_config=generation_config
        )

        formatted_history = []
        for msg in chat_history:
            role = 'user' if msg['role'] == 'user' else 'model'
            formatted_history.append({'role': role, 'parts': [msg['content'][:800]]})

        chat = gemini_model.start_chat(history=formatted_history)

        # IMPORTANT: send payload, not raw user text
        response = chat.send_message(
            json.dumps(payload, separators=(",", ":")),
            stream=True
        )

        for chunk in response:
            if chunk.text:
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

# SM4 Search API route
@app.route("/api/search", methods=["POST"])
def api_search():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file type (png/jpg/jpeg/webp only)"}), 400

    # Save upload (keep your existing behavior so your UI can show query image from /static/uploads)
    safe_name = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{ts}_{safe_name}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(saved_path)

    # Also write to a temp file for gradio_client (path-based input is most reliable)
    suffix = os.path.splitext(saved_name)[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        # copy bytes into temp file
        with open(saved_path, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())

        # Call HF Space (Gradio API)
        client = get_sm4_client()
        result = client.predict(
            img=handle_file(tmp_path),
            api_name="/run_search"
        )

        # HF returns tuple: (query_img_dict, prediction_text, gallery_list, explanation_md)
        query_img = result[0] or {}
        pred_class = result[1] or ""
        gallery = result[2] or []
        explanation_md = result[3] or ""

        # Convert gallery to "results" list (best-effort, since HF does not return Product_ID/score)
        results_list = []
        for i, item in enumerate(gallery, start=1):
            # Usually: {"image": {...}, "caption": "..."}
            image_obj = item.get("image") if isinstance(item, dict) else None
            caption = item.get("caption") if isinstance(item, dict) else None

            image_url = image_obj.get("url") if isinstance(image_obj, dict) else None
            image_path = image_obj.get("path") if isinstance(image_obj, dict) else None

            # Keep keys your existing UI may rely on
            results_list.append({
                "Product_ID": f"hf_{i}",                 # placeholder (HF doesn’t provide your catalog IDs)
                "Product_Title": caption or f"Result {i}",
                "image_url": image_url,                  # use if present (browser-loadable)
                "image_path": image_path,                # usually NOT browser-loadable
                "similarity_score": None,                # HF output doesn't guarantee scores
                "match_label": None,
                "match_badge_class": None,
                "ai_summary": ""                         # you can optionally fill from explanation_md if you want
            })

        # Since HF doesn't provide a confidence score, keep these neutral
        top1_score = None
        low_conf = False

        # Keep fields so your frontend doesn't break
        query_desc = {"query_caption": "", "query_tags": []}

        # Treat HF markdown as the AI overview (since HF already generated it)
        ai = {"overview": explanation_md, "detected_features": [], "item_summaries": {}}

        return jsonify({
            "ok": True,
            # Keep your local upload URL so the UI can always show the query image
            "query_image_url": f"/static/uploads/{saved_name}",
            "query_filename": saved_name,

            # Keep same key name as before (your UI expects pred_class)
            "pred_class": pred_class,

            # Use HF-generated markdown here
            "ai_overview": ai.get("overview"),
            "detected_features": ai.get("detected_features", []),

            # Top-5 results derived from HF gallery
            "results": results_list,

            "top1_score": top1_score,
            "low_confidence": low_conf,

            "query_caption": query_desc.get("query_caption", ""),
            "query_tags": query_desc.get("query_tags", []),

            # Optional: expose HF query url if you want (not required)
            # "hf_query_image_url": query_img.get("url")
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)