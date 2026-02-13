import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MODEL_PATH = os.path.join(BASE_DIR, "attribute_lr_best_pipeline.joblib")
    CATALOG_PATH = os.path.join(BASE_DIR, "fully_cleaned_catalog.csv")
    
    # API Keys
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    HF_API_URL = os.environ.get("HF_API_URL") # Hugging Face Space URL
    HF_TOKEN = os.environ.get("HF_TOKEN")     # Optional HF Token
    
    # App Settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 # 10MB
    DEBUG = True
    PORT = 5000