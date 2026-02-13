import pandas as pd
import os
import json
import google.generativeai as genai
from config import Config

class PrinterRecommender:
    def __init__(self):
        self.catalog = None
        self.load_catalog()
        self.setup_gemini()

    def load_catalog(self):
        if os.path.exists(Config.CATALOG_PATH):
            try:
                self.catalog = pd.read_csv(Config.CATALOG_PATH)
                # Ensure we only recommend from 'Printer' category
                self.catalog = self.catalog[self.catalog['category'] == 'Printer']
                print(f"Recommender catalog loaded: {len(self.catalog)} printers.")
            except Exception as e:
                print(f"Error loading catalog: {e}")

    def setup_gemini(self):
        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)

    def get_top_matches(self, user_query, predicted_attributes):
        if self.catalog is None:
            return []

        df = self.catalog.copy()
        
        # 1. Attribute Filtering (SM2 Logic)
        # Assuming predicted_attributes is [HomeUse, Wireless, Duplex, BudgetConcern]
        if len(predicted_attributes) >= 4:
            if predicted_attributes[3] == 1: # BudgetConcern
                df = df[df['price_cleaned'].str.replace('$', '').str.replace(',', '').astype(float) < 300]
            
            if predicted_attributes[1] == 1: # Wireless
                df = df[df['details_cleaned'].str.contains('Wireless|WiFi|Wi-Fi', case=False, na=False)]
                
            if predicted_attributes[2] == 1: # Duplex
                df = df[df['details_cleaned'].str.contains('Duplex|Two-sided', case=False, na=False)]

        # 2. Keyword Search (Simple BM25-like)
        keywords = user_query.lower().split()
        def score_row(row):
            score = 0
            text = f"{row['title']} {row['details_cleaned']}".lower()
            for kw in keywords:
                if kw in text:
                    score += 1
            return score

        df['search_score'] = df.apply(score_row, axis=1)
        top_matches = df.sort_values(by='search_score', ascending=False).head(5)
        
        return top_matches[['title', 'price_cleaned', 'details_cleaned', 'image_url']].to_dict(orient='records')

    def generate_stream(self, user_input, predicted_attributes, chat_history):
        if not Config.GEMINI_API_KEY:
            yield f"data: {json.dumps({'error': 'Gemini API Key missing'})}\n\n"
            return

        # Get only the most relevant products to save tokens
        matches = self.get_top_matches(user_input, predicted_attributes)
        context = json.dumps(matches, indent=2)
        
        system_instruction = f"""
        You are a fast, professional Printer Expert for Singink.
        
        CONTEXT (Top 5 Matches):
        {context}
        
        USER INTENT HINTS (from classifier): {predicted_attributes}
        
        INSTRUCTIONS:
        1. If the user's request is vague, ask ONE targeted follow-up question.
        2. If clear, recommend 2-3 printers from the CONTEXT above.
        3. For each recommendation, include the image: ![Product Image](IMAGE_URL)
        4. Explain WHY based on features.
        5. Be concise to ensure fast streaming.
        """
        
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system_instruction)
            formatted_history = [{'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']]} for m in chat_history]
            chat = model.start_chat(history=formatted_history)
            
            response = chat.send_message(user_input, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"