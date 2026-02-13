import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from datetime import datetime

from config import Config
from classifier import IntentClassifier
from model_deployment.visual_search import VisualSearchClient
from recommender import PrinterRecommender

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Modules
classifier = IntentClassifier()
visual_search = VisualSearchClient()
recommender = PrinterRecommender()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_history = data.get('history', [])
    
    # 1. Classify Intent (SM2)
    predicted_attributes = classifier.predict(user_message)
    
    # 2. Generate Streaming Response (Optimized Recommender)
    return Response(
        stream_with_context(recommender.generate_stream(user_message, predicted_attributes, chat_history)),
        mimetype='text/event-stream'
    )

@app.route("/api/search", methods=["POST"])
def api_search():
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400

    safe_name = secure_filename(file.filename)
    saved_path = os.path.join(Config.UPLOAD_FOLDER, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}")
    file.save(saved_path)

    # 3. Visual Search (SM4 via Hugging Face)
    result = visual_search.search(saved_path)
    
    if result["ok"]:
        return jsonify({
            "ok": True,
            "query_image_url": f"/static/uploads/{os.path.basename(saved_path)}",
            "prediction": result["prediction"],
            "results": result["gallery"],
            "ai_text": result["ai_text"]
        })
    else:
        return jsonify({"ok": False, "error": result["error"]}), 500

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, port=Config.PORT)