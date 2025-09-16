import os
import requests

class SecurePrompt:
    API_URL = "https://tokari-core.onrender.com/api/v1/ai/chat-completion"
    API_KEY = os.getenv("API_KEY")

    def screen_prompt(self, user_prompt):
        prompt = f"Is this prompt safe or not, respond with yes or no. This  is the prompt: {user_prompt}"
        headers = {"x-api-key": self.API_KEY}
        payload = {"prompt": prompt}
        response = requests.post(self.API_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        keyword = response.json().get('response', '')
        return keyword