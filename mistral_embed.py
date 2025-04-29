import os
from typing import List
import requests

class MistralEmbed:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": "mistral-embed", "input": text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]