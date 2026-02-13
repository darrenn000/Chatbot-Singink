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
            # Call the 'run_search' function in the Gradio app
            # Based on SM4-main/app.py, the function is 'run_search'
            # We use handle_file for local paths
            result = self.client.predict(
                img=handle_file(image_path),
                api_name="/run_search"
            )
            
            # Gradio returns: (img, prediction_text, gallery_data, ai_text)
            # We need to parse this for our UI
            return {
                "ok": True,
                "prediction": result[1],
                "gallery": result[2], # List of (image_path, caption)
                "ai_text": result[3]
            }
        except Exception as e:
            print(f"Visual search error: {e}")
            return {"ok": False, "error": str(e)}
