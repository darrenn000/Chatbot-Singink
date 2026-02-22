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
import shutil
import uuid

# Load environment variables from .env file if it exists
load_dotenv()

# ✅ DEBUG: confirm which file is actually running
print("[BOOT] running:", __file__)

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "fully_cleaned_catalog.csv")
MODEL_PATH = os.path.join(BASE_DIR, "attribute_lr_best_pipeline.joblib")

DEBUG_ROUTER = os.environ.get("DEBUG_ROUTER", "0") == "1"

# SM4 Space configuration
SM4_SPACE_ID = os.environ.get("SM4_SPACE_ID", "lirou-0/sm4_visual_similarity")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only if space is private
sm4_client = None

# =====================================================================================
# SM01 (Intent Classifier) — ROUTER
SM01_SPACE_ID = os.environ.get("SM01_SPACE_ID", "Diabvell/Pure_Intent_Classifier")
sm01_client = None

def get_sm01_client():
    global sm01_client
    if sm01_client is None:
        sm01_client = Client(SM01_SPACE_ID, hf_token=HF_TOKEN) if HF_TOKEN else Client(SM01_SPACE_ID)
    return sm01_client

def sm01_predict_intent(text: str) -> dict:
    """
    Returns: {"intent": str, "confidence": float, "label_id": int}
    """
    c = get_sm01_client()
    return c.predict(message=text, api_name="/intent_only")
# =====================================================================================

def get_sm4_client():
    global sm4_client
    if sm4_client is None:
        sm4_client = Client(SM4_SPACE_ID, hf_token=HF_TOKEN) if HF_TOKEN else Client(SM4_SPACE_ID)
    return sm4_client

# =====================================================================================
# ✅ SM3 (Your Sentiment Analysis Space) — Integration
# Space: gracewidj/sm3-sentiment-analysis
# API endpoint from your screenshot: /submit_and_accumulate
SM3_SPACE_ID = os.environ.get("SM3_SPACE_ID", "gracewidj/sm3-sentiment-analysis")
sm3_client = None

def get_sm3_client():
    """Always return a FRESH Gradio client session.

    Your SM3 HuggingFace Space keeps accumulator state per client session.
    If we reuse a single Client, its server-side state can leak across printers and
    cause duplicates (e.g., 10 + 5 + 10). Creating a new Client each call isolates
    state and makes 'clear' behave predictably.
    """
    return Client(SM3_SPACE_ID, hf_token=HF_TOKEN) if HF_TOKEN else Client(SM3_SPACE_ID)

# --- SM3 per-printer storage (server-side) ---
# In-memory store: {printer_id: [review1, review2, ...]}
SM3_STORE = {}

def _normalize_printer_id(pid: str) -> str:
    pid = (pid or "").strip()
    return pid if pid else "general"

def _split_reviews_text(text: str) -> list:
    return [r.strip() for r in (text or "").splitlines() if r.strip()]

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ✅ NEW: clear HF Space history so it doesn't keep global accumulated state
def sm3_space_clear_history() -> bool:
    """
    Best-effort: tries common Gradio api_names for clearing/resetting state.
    If your Space has a dedicated 'Clear' button, it usually maps to one of these.
    """
    c = get_sm3_client()
    api_candidates = [
        "/clear_history",
        "/clear",
        "/reset",
        "/reset_history",
        "/clear_and_reset",
        "/clear_state",
    ]

    # try calling with no args and with a dummy arg (some spaces require at least 1 input)
    call_variants = [
        lambda api: c.predict(api_name=api),
        lambda api: c.predict("", api_name=api),
        lambda api: c.predict(new_text="", api_name=api),
    ]

    for api in api_candidates:
        for call in call_variants:
            try:
                call(api)
                return True
            except Exception:
                pass
    return False

def unpack_sm3_result(result):
    """
    Attempts to unpack Gradio outputs from SM3 Space.

    Expected-ish outputs (varies by Space):
    - result could be list/tuple: [overall_text, table, ...more strings/markdown]
    - table is often dict: {"headers":[...], "data":[[...], ...]}
    """
    overall = ""
    rows = []
    extras_texts = []

    if result is None:
        return overall, rows, ""

    if isinstance(result, (list, tuple)):
        if len(result) >= 1 and result[0] is not None:
            overall = str(result[0])

        # find first "table-like" object
        table_obj = None
        for x in result:
            if isinstance(x, dict) and "headers" in x and "data" in x:
                table_obj = x
                break
            if isinstance(x, list) and x and isinstance(x[0], dict):
                table_obj = x
                break

        if table_obj is not None:
            if isinstance(table_obj, dict):
                headers = table_obj.get("headers") or []
                data = table_obj.get("data") or []
                # map indices safely
                def idx(name, fallback):
                    try:
                        return headers.index(name)
                    except Exception:
                        return fallback

                i_review = idx("review", 0)
                i_sent = idx("sentiment", 1)
                i_conf = idx("confidence", 2)

                for r in data:
                    if not isinstance(r, (list, tuple)) or len(r) == 0:
                        continue
                    rows.append({
                        "review": r[i_review] if i_review < len(r) else "",
                        "sentiment": r[i_sent] if i_sent < len(r) else "",
                        "confidence": _safe_float(r[i_conf] if i_conf < len(r) else 0.0),
                    })

            elif isinstance(table_obj, list) and table_obj and isinstance(table_obj[0], dict):
                # list-of-dicts rows
                for rr in table_obj:
                    rows.append({
                        "review": rr.get("review", ""),
                        "sentiment": rr.get("sentiment", ""),
                        "confidence": _safe_float(rr.get("confidence", 0.0)),
                    })

        # collect any extra text outputs (markdown / insights) WITHOUT dumping reviews wall
        for x in result:
            if isinstance(x, str) and x.strip():
                extras_texts.append(x.strip())

        insights = "\n\n".join(extras_texts[-2:]) if extras_texts else ""
        return overall, rows, insights

    if isinstance(result, dict):
        return "", [], json.dumps(result, indent=2)

    return "", [], str(result)

# --- SM3 Themes / Feature Ratings (keyword-based) ---
FEATURE_KEYWORDS = {
    "Print Quality": [r"\bprint quality\b", r"\bsharp\b", r"\bclear\b", r"\bcrisp\b", r"\bsmudge\b", r"\bblurry\b"],
    "Printing Speed": [r"\bfast\b", r"\bspeed\b", r"\bslow\b", r"\bfirst page\b"],
    "Setup Ease": [r"\bsetup\b", r"\binstallation\b", r"\beasy to install\b", r"\bconfigure\b"],
    "Value for Money": [r"\bvalue\b", r"\bworth\b", r"\bprice\b", r"\bexpensive\b", r"\bcheap\b"],
    "Reliability": [r"\breliable\b", r"\bconsistent\b", r"\bstopped working\b", r"\bbroke\b", r"\bfaulty\b"],
    "Paper Jams": [r"\bpaper jam\b", r"\bjams\b", r"\bstuck\b"],
    "User-Friendliness": [r"\buser[- ]friendly\b", r"\beasy to use\b", r"\bsimple controls\b"],
    "Noise Level": [r"\bnoisy\b", r"\bloud\b", r"\bnoise\b", r"\bvibrat"],
    "Wi-Fi Connectivity": [r"\bwifi\b", r"\bwi-fi\b", r"\bwireless\b", r"\bdisconnect\b", r"\bconnection drops\b"],
    "Toner/Ink Cost": [r"\btoner\b", r"\bink\b", r"\bcartridge\b", r"\bcost\b", r"\bexpensive\b"],
}

def _review_mentions_feature(review: str, patterns: list) -> bool:
    s = (review or "").lower()
    for p in patterns:
        if re.search(p, s, re.IGNORECASE):
            return True
    return False

def build_sm3_aggregates(rows: list):
    """
    rows: [{"review":..., "sentiment":..., "confidence":...}, ...]
    Returns:
      - love_themes: list
      - concern_themes: list
      - feature_ratings: list
    """
    # feature stats
    stats = {}
    for feature, pats in FEATURE_KEYWORDS.items():
        stats[feature] = {
            "feature": feature,
            "mentions": 0,
            "pos": 0,
            "neg": 0,
            "conf_sum": 0.0,
        }

    for r in rows:
        txt = str(r.get("review", "") or "")
        sent = str(r.get("sentiment", "") or "").lower()
        conf = _safe_float(r.get("confidence", 0.0))
        for feature, pats in FEATURE_KEYWORDS.items():
            if _review_mentions_feature(txt, pats):
                stats[feature]["mentions"] += 1
                stats[feature]["conf_sum"] += conf
                if sent == "positive":
                    stats[feature]["pos"] += 1
                elif sent == "negative":
                    stats[feature]["neg"] += 1

    feature_ratings = []
    for feature, d in stats.items():
        m = d["mentions"]
        if m <= 0:
            continue
        pos_pct = (d["pos"] / m) * 100.0
        neg_pct = (d["neg"] / m) * 100.0
        avg_conf = d["conf_sum"] / m if m else 0.0

        # rating 1..5 based on positive percentage (simple + understandable)
        rating_1to5 = round(1.0 + 4.0 * (pos_pct / 100.0), 2)

        feature_ratings.append({
            "feature": feature,
            "mentions": m,
            "positive_%": round(pos_pct, 1),
            "negative_%": round(neg_pct, 1),
            "avg_conf": round(avg_conf, 3),
            "rating_1to5": rating_1to5
        })

    # "What users love": themes with highest positive_% (and mentions as tie-break)
    love_themes = sorted(
        [{"theme": x["feature"], "positive_%": x["positive_%"], "mentions": x["mentions"]} for x in feature_ratings],
        key=lambda t: (t["positive_%"], t["mentions"]),
        reverse=True
    )[:6]

    # "Common concerns": themes with highest negative_%
    concern_themes = sorted(
        [{"theme": x["feature"], "negative_%": x["negative_%"], "mentions": x["mentions"]} for x in feature_ratings],
        key=lambda t: (t["negative_%"], t["mentions"]),
        reverse=True
    )[:6]

    # Sort full table in a stable useful way (mentions desc, then rating desc)
    feature_ratings = sorted(feature_ratings, key=lambda x: (x["mentions"], x["rating_1to5"]), reverse=True)

    return love_themes, concern_themes, feature_ratings

def sm3_call_space(reviews_list: list):
    """
    Calls your SM3 Space with newline-joined reviews.
    Prefer NON-ACCUMULATING endpoints (so no duplicates / no global memory leaks).
    Fallback to /submit_and_accumulate only if needed.
    """
    c = get_sm3_client()
    joined = "\n".join([r.strip() for r in reviews_list if str(r).strip()])

    # try safer endpoints first
    api_candidates = [
        "/analyze",
        "/predict",
        "/classify",
        "/run",
        "/inference",
        "/analyze_text",
        "/sentiment",
        "/process",
    ]

    # common argument names spaces use
    call_variants = [
        lambda api: c.predict(joined, api_name=api),
        lambda api: c.predict(text=joined, api_name=api),
        lambda api: c.predict(new_text=joined, api_name=api),
        lambda api: c.predict(message=joined, api_name=api),
    ]

    for api in api_candidates:
        for call in call_variants:
            try:
                return call(api)
            except Exception:
                pass

    # last resort: accumulating endpoint
    return c.predict(new_text=joined, api_name="/submit_and_accumulate")

# =====================================================================================

# --- SM4 Upload Configuration ---
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

# =====================================================================================
# SM01 -> SM02 ROUTER TABLE (Intent gating)
PRODUCT_INTENTS = {"product_search", "product_information", "price_check", "stock_check"}

def non_product_response(intent: str, confidence: float):
    """
    Non-product intents: return immediately (NO SM02, NO Gemini).
    """
    if intent == "return_refund":
        return {
            "intent": intent,
            "confidence": confidence,
            "message": "For returns/refunds, please head to the FAQ to resolve your query."
        }

    if intent == "promo_discount":
        return {
            "intent": intent,
            "confidence": confidence,
            "message": "For promotions/discounts, please head to the FAQ to resolve your query."
        }

    # Fallback
    return {
        "intent": intent,
        "confidence": confidence,
        "message": "Please head to the FAQ to resolve your query."
    }

def override_intent(user_message: str, intent: str) -> str:
    """
    Guardrails to prevent obvious promo/refund messages from being misrouted as product_search.
    """
    msg = (user_message or "").lower()

    # refund / return keywords
    if any(w in msg for w in ["refund", "return", "exchange", "damaged", "broken", "defective", "faulty"]):
        return "return_refund"

    # promo / discount keywords
    if any(w in msg for w in ["promo", "promotion", "discount", "voucher", "coupon", "code", "deal", "sale"]):
        return "promo_discount"

    return intent
# =====================================================================================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def select_candidates(user_message: str, catalog_df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """
    Cheap Top-K retrieval to avoid sending the full catalog to Gemini.
    """
    df = catalog_df[catalog_df["category"] == "Printer"].copy()
    msg = _norm(user_message)

    # Heuristic boosts
    if any(w in msg for w in ["wireless", "wifi", "wi-fi"]):
        df["_wireless"] = df["details_cleaned"].fillna("").str.lower().str.contains("wifi|wi-fi|wireless")
        df = df.sort_values("_wireless", ascending=False)

    if any(w in msg for w in ["duplex", "double-sided", "double sided"]):
        df["_duplex"] = df["details_cleaned"].fillna("").str.lower().str.contains("duplex|double[- ]sided")
        df = df.sort_values("_duplex", ascending=False)

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
            "details": str(r.get("details_cleaned", ""))[:260],
            "image_url": r.get("image_url", ""),
        })
    return packed

def generate_streaming_response(user_input, predicted_attributes, chat_history):
    """
    Generator function that yields chunks of text from Gemini.
    """
    if not api_key:
        yield "data: " + json.dumps({"error": "API Key is missing"}) + "\n\n"
        return

    if printer_catalog is None or printer_catalog.empty:
        yield "data: " + json.dumps({"error": "Catalog not loaded"}) + "\n\n"
        return

    chat_history = (chat_history or [])[-4:]

    candidates_df = select_candidates(user_input, printer_catalog, k=12)
    candidates = pack_products(candidates_df)

    system_instruction = (
        "You are a Printer Expert Assistant.\n"
        "You will receive a JSON payload with:\n"
        "- user_message\n"
        "- predicted_attributes: {intent, intent_confidence, attributes}\n"
        "- candidates: a list of printers with title, price, details, image_url\n\n"
        "STRICT RULES:\n"
        "1) Follow the intent in predicted_attributes.intent.\n"
        "2) Only use information from candidates. Do NOT invent models or specs.\n"
        "3) If you cannot find an exact model name in candidates, say so and offer the closest matches from candidates.\n"
        "4) Keep the response concise.\n\n"
        "INTENT BEHAVIOR:\n"
        "- product_search: If key info is missing (budget, must-have features), ask exactly ONE short follow-up question.\n"
        "  Otherwise recommend exactly 2 printers from candidates.\n"
        "- price_check: Recommend exactly 2 printers from candidates and include price for each (if available). Do NOT ask budget.\n"
        "- product_information: If a specific model name is mentioned but not present in candidates, say it's not found and show 2 closest matches.\n"
        "  If present, pick the closest 1 printer and summarize key features from details.\n"
        "- stock_check: Treat 'available' as 'found in our catalog/listing'.\n"
        "  If the exact model appears in candidates, say it is available and show that exact model first.\n"
        "  If the model does NOT appear in candidates, say it is not available and show 2 closest matches.\n\n"
        "OUTPUT FORMAT:\n"
        "- If asking a question: output ONLY the single question.\n"
        "- If recommending: for each product include:\n"
        "#### {Product_Title}\n"
        "  ![Product Image](IMAGE_URL)\n"
        "**Price:** ${Price}\n\n"
        "  **Why it matches:** One short sentence.\n\n"
        "  **Key Specifications:**\n"
        "  - Spec 1\n"
        "  - Spec 2\n"
        "  - Spec 3\n"
    )

    payload = {
        "user_message": user_input[:800],
        "predicted_attributes": predicted_attributes,
        "candidates": candidates
    }

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
            role = 'user' if msg.get('role') == 'user' else 'model'
            formatted_history.append({'role': role, 'parts': [str(msg.get('content', ''))[:800]]})

        chat = gemini_model.start_chat(history=formatted_history)

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

def sse_one_message(text: str):
    # One-shot SSE message compatible with your existing frontend
    yield f"data: {json.dumps({'text': text})}\n\n"

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_message = (data.get('message', '') or '').strip()
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # ✅ SM3 trigger (simple + clear):
    # User types: "sm3: <reviews...>" (multi-line supported)
    low = user_message.lower()
    if low.startswith("sm3:") or low.startswith("/sm3"):
        reviews_text = user_message.split(":", 1)[1] if ":" in user_message else ""
        reviews = _split_reviews_text(reviews_text)
        if not reviews:
            msg = "Please provide reviews (one per line) after `sm3:`"
        else:
            # ✅ FIX: one-shot analyze only (DO NOT SAVE into SM3_STORE / general)
            sm3_space_clear_history()

            result = sm3_call_space(reviews)
            overall, rows, insights = unpack_sm3_result(result)
            love, concerns, feature_ratings = build_sm3_aggregates(rows)

            msg = "### ✅ SM3 Sentiment (one-shot)\n\n"
            msg += f"{overall}\n\n"
            msg += f"Per-review rows: {len(rows)}\n\n"
            msg += "Use the UI button to save per-printer + see full tables."

        return Response(stream_with_context(sse_one_message(msg)), mimetype="text/event-stream")

    # =================================================================================
    # (SM01) Intent classification (priority)
    try:
        intent_payload = sm01_predict_intent(user_message)
        intent = intent_payload.get("intent")
        intent_conf = float(intent_payload.get("confidence", 0.0))
    except Exception as e:
        print(f"[SM01] intent call failed: {e}")
        intent, intent_conf = "product_search", 0.0

    # Guardrail override for obvious promo/refund queries
    intent = override_intent(user_message, intent)

    if DEBUG_ROUTER:
        print("[SM01 RESULT]", intent, intent_conf)
        print("[ROUTE CHECK] intent=", intent, "PRODUCT_INTENTS=", PRODUCT_INTENTS)
    # =================================================================================

    # ✅ NON-PRODUCT: return one-shot SSE (frontend expects SSE)
    if intent not in PRODUCT_INTENTS:
        msg = non_product_response(intent, intent_conf).get(
            "message",
            "Please head to the FAQ to resolve your query."
        )
        return Response(
            stream_with_context(sse_one_message(msg)),
            mimetype="text/event-stream"
        )

    # ✅ PRODUCT: continue normal SM02 + Gemini streaming
    attrs = []
    if model:
        try:
            prediction = model.predict([user_message])
            attrs = prediction.tolist()
        except Exception as e:
            print(f"Prediction error: {e}")

    predicted_attributes = {
        "intent": intent,
        "intent_confidence": intent_conf,
        "attributes": attrs,
    }

    return Response(
        stream_with_context(generate_streaming_response(user_message, predicted_attributes, chat_history)),
        mimetype='text/event-stream'
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "classifier_loaded": model is not None,
        "catalog_loaded": printer_catalog is not None,
        "sm3_space": SM3_SPACE_ID
    })

# ✅ NEW: expose printer list for SM3 dropdown (doesn't touch SM1/2/4 logic)
@app.route("/api/printers", methods=["GET"])
def api_printers():
    items = []
    if printer_catalog is not None and not printer_catalog.empty:
        df = printer_catalog
        if "category" in df.columns:
            df = df[df["category"] == "Printer"]
        # safest field: title
        if "title" in df.columns:
            titles = df["title"].fillna("").astype(str).tolist()
            # unique + keep order
            seen = set()
            for t in titles:
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    items.append(t)
    # always include General
    return jsonify({"ok": True, "printers": ["general"] + items})

load_resources()

# =====================================================================================
# ✅ SM3 API route (ONLY SM3 CHANGED)
# Supports:
# - action="submit" (append reviews then analyze)
# - action="analyze" (analyze existing saved reviews)
# - action="clear" (clear saved reviews for that printer)
# - action="status" (return counts)
@app.route("/api/sm3", methods=["POST"])
def api_sm3():
    data = request.get_json(silent=True) or {}

    action = (data.get("action") or "submit").strip().lower()
    printer_id = _normalize_printer_id(data.get("printer_id"))

    # accept:
    # - reviews: multiline string
    # - reviews: list[str]
    raw_reviews = data.get("reviews", [])
    reviews_list = []

    if isinstance(raw_reviews, str):
        reviews_list = _split_reviews_text(raw_reviews)
    elif isinstance(raw_reviews, list):
        reviews_list = [str(r).strip() for r in raw_reviews if str(r).strip()]

    # init bucket
    SM3_STORE.setdefault(printer_id, [])

    if action == "clear":
        SM3_STORE[printer_id] = []
        # ✅ ALSO clear the HF Space accumulator so it stops returning old stuff (best-effort)
        sm3_space_clear_history()
        return jsonify({
            "ok": True,
            "printer_id": printer_id,
            "cleared": True,
            "saved_count": 0
        })

    if action == "status":
        return jsonify({
            "ok": True,
            "printer_id": printer_id,
            "saved_count": len(SM3_STORE.get(printer_id, []))
        })

    if action == "submit":
        if not reviews_list:
            return jsonify({"ok": False, "error": "No reviews provided"}), 400
        SM3_STORE[printer_id].extend(reviews_list)

    # action == "analyze" OR after submit:
    saved = SM3_STORE.get(printer_id, [])
    if not saved:
        return jsonify({"ok": False, "error": "No saved reviews for this printer"}), 400

    try:
        # ✅ Best-effort clear remote accumulator (still fine if no endpoint exists)
        sm3_space_clear_history()

        result = sm3_call_space(saved)
        overall, rows, insights = unpack_sm3_result(result)
        love, concerns, feature_ratings = build_sm3_aggregates(rows)

        # IMPORTANT: do NOT dump all reviews text. Only provide summary + tables.
        return jsonify({
            "ok": True,
            "printer_id": printer_id,
            "saved_count": len(saved),

            # summary
            "overall": overall,

            # tables
            "rows": rows,  # per-review results
            "what_users_love": love,
            "common_concerns": concerns,
            "feature_ratings": feature_ratings,

            # keep insights if your HF space returns something useful, but frontend can ignore if empty
            "insights": insights
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# =====================================================================================
def slugify(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\n.*$", "", s)           # remove score line
    s = re.sub(r"[^a-z0-9]+", "_", s)     # keep alnum, replace others with _
    s = s.strip("_")
    return s

STATIC_IMAGE_DIR = os.path.join(BASE_DIR, "static", "image")

def build_static_image_index():
    idx = {}
    if not os.path.isdir(STATIC_IMAGE_DIR):
        return idx
    for fn in os.listdir(STATIC_IMAGE_DIR):
        base, ext = os.path.splitext(fn)
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue
        idx[base.lower()] = fn
    return idx

STATIC_IMAGE_INDEX = build_static_image_index()
SM4_THUMBS_DIR = os.path.join(BASE_DIR, "static", "sm4_thumbs")
os.makedirs(SM4_THUMBS_DIR, exist_ok=True)
# SM4 Search API route 
@app.route("/api/search", methods=["POST"])
def api_search():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file type (png/jpg/jpeg/webp only)"}), 400

    safe_name = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{ts}_{safe_name}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(saved_path)

    suffix = os.path.splitext(saved_name)[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        with open(saved_path, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())

        client = get_sm4_client()
        result = client.predict(
            img=handle_file(tmp_path),
            api_name="/run_search"
        )

        query_img = result[0] or {}
        pred_class = result[1] or ""
        gallery = result[2] or []
        explanation_md = result[3] or ""

        results_list = []
        for i, item in enumerate(gallery, start=1):
            caption = item.get("caption") if isinstance(item, dict) else None
            image_obj = item.get("image") if isinstance(item, dict) else None

            image_url = None
            image_path = None

            # SM4 returns a local path string like C:\Users\...\Temp\gradio\...\image.webp
            if isinstance(image_obj, str):
                image_path = image_obj

                # If it exists, copy it into static so the browser can load it
                if os.path.exists(image_path):
                    ext = os.path.splitext(image_path)[1] or ".webp"
                    out_name = f"sm4_{uuid.uuid4().hex}{ext}"
                    out_path = os.path.join(SM4_THUMBS_DIR, out_name)

                    shutil.copyfile(image_path, out_path)

                    # Browser-loadable URL
                    image_url = f"/static/sm4_thumbs/{out_name}"

            results_list.append({
                "Product_ID": f"hf_{i}",
                "Product_Title": caption or f"Result {i}",
                "image_url": image_url,
                "image_path": image_path,
                "similarity_score": None,
                "match_label": None,
                "match_badge_class": None,
                "ai_summary": ""
            })

        top1_score = None
        low_conf = False
        query_desc = {"query_caption": "", "query_tags": []}
        ai = {"overview": explanation_md, "detected_features": [], "item_summaries": {}}

        return jsonify({
            "ok": True,
            "query_image_url": f"/static/uploads/{saved_name}",
            "query_filename": saved_name,
            "pred_class": pred_class,
            "ai_overview": ai.get("overview"),
            "detected_features": ai.get("detected_features", []),
            "results": results_list,
            "top1_score": top1_score,
            "low_confidence": low_conf,
            "query_caption": query_desc.get("query_caption", ""),
            "query_tags": query_desc.get("query_tags", []),
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