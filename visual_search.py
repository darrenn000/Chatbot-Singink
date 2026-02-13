import os
from gradio_client import Client, handle_file
from config import Config

class VisualSearchClient:
    def __init__(self):
        self.client = None
        self.api_url = Config.HF_API_URL
        self.token = Config.HF_TOKEN
        self.init_client()

    def init_client(self):
        if self.api_url:
            try:
                # Connect to the Gradio Space
                self.client = Client(self.api_url, hf_token=self.token)
                print(f"Connected to SM4 Visual Search at {self.api_url}")
            except Exception as e:
                print(f"Error connecting to HF Space: {e}")
        else:
            print("HF_API_URL not configured. Visual search disabled.")

    def search(self, image_path):
        if not self.client:
            return {"ok": False, "error": "Visual search client not initialized."}

        try:
            # We try the most common Gradio API names for gr.Interface
            # 1. Try '/predict'
            # 2. If that fails, the client will automatically try the default function
            result = self.client.predict(
                handle_file(image_path),
                api_name="/predict" 
            )
            
            return {
                "ok": True,
                "prediction": result[1],
                "gallery": result[2],
                "ai_text": result[3]
            }
        except Exception as e:
            # Fallback: Try without an explicit api_name if /predict fails
            try:
                result = self.client.predict(handle_file(image_path))
                return {
                    "ok": True,
                    "prediction": result[1],
                    "gallery": result[2],
                    "ai_text": result[3]
                }
            except Exception as e2:
                return {"ok": False, "error": f"Gradio Error: {str(e2)}"}
