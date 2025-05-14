from typing import List, Dict, Optional
import aiohttp
import asyncio
import json
import os

import requests

api_key = os.getenv('groq_key')

class SmartAgentSystem:
    def __init__(self):
        api_key = os.getenv('groq_key')
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def expert_agent(self, text: str) -> Optional[Dict]:
        """Agente especialista com identificação de chunk"""
        
        prompt = """
            You are an assistant specialized in sentiment analysis of restaurant reviews.\n
            Classify the sentiment of the user-provided comment into one of the following 
            categories: \"Positive\", \"Negative\", or \"Neutral\". Bring related metadata 
            if available.\nStrictly respond with a JSON object containing only the key \"sentiment\" and 
            the category value, and \"metadata\" with the appropriate metadata (if any), including \"responsible\" 
            and \"reason\".\nExample: {\"sentiment\": \"Positive\", \"metadata\": {\"responsible\": \"waiter\", \"reason\":
            \"attentive service\"}}\nIf the comment does not clearly express a sentiment (e.g., 
            it only describes something without opinion), classify it as \"Neutral\".\nIf the comment 
            expresses mixed sentiments (e.g., the food was good, but the service was bad), 
            try to determine the predominant overall sentiment or use \"Neutral\" if it is balanced.
        """
        
        try:
            async with self.session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "messages": [{
                        "role": "system",
                        "content": prompt
                    }, {
                        "role": "user",
                        "content": text
                    }],
                    "temperature": 0.5,
                    "response_format": {"type": "json_object"}
                },
                timeout=20
            ) as response:
                response.raise_for_status()
                data = await response.json()
                result = json.loads(data['choices'][0]['message']['content'])
                
                print(f"Expert Agent Decision: {result}")
                
                # Validate response structure
                required_keys = {'sentiment'}
                if not all(key in result for key in required_keys):
                    raise ValueError("Missing required keys in response")
                
                return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Error: {str(e)}")
        except KeyError as e:
            print(f"Missing key response: {str(e)}")
        except Exception as e:
            print(f"Error processing: {str(e)}")
        return None
    
