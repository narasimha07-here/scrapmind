import openai
import requests
import json
from typing import Dict, List, Optional
import streamlit as st

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://your-chatbot-app.com",
                "X-Title": "No-Code Chatbot Builder"
            }
        )
    
    def get_available_models(self) -> List[Dict]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-chatbot-app.com",
                "X-Title": "No-Code Chatbot Builder"
            }
            
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models_data = response.json()
                models = []
                
                for model in models_data.get('data', []):
                    models.append({
                        'id': model['id'],
                        'name': model.get('name', model['id']),
                        'description': model.get('description', ''),
                        'pricing': model.get('pricing', {}),
                        'context_length': model.get('context_length', 4096),
                        'top_provider': model.get('top_provider', {}),
                        'is_free': 'free' in model['id'].lower()
                    })
                
                models.sort(key=lambda x: (not x['is_free'], x['name']))
                return models
            else:
                return self._get_fallback_models()
                
        except Exception as e:
            st.warning(f"Error fetching models: {str(e)}. Using fallback models.")
            return self._get_fallback_models()
    
    def _get_fallback_models(self) -> List[Dict]:
        return [
            {"id": "meta-llama/llama-3.2-3b-instruct:free", "name": "Llama 3.2 3B (Free)", "description": "Free fast model", "is_free": True},
            {"id": "meta-llama/llama-3.2-1b-instruct:free", "name": "Llama 3.2 1B (Free)", "description": "Free lightweight model", "is_free": True},
            {"id": "huggingfaceh4/zephyr-7b-beta:free", "name": "Zephyr 7B Beta (Free)", "description": "Free conversational model", "is_free": True},
            {"id": "openchat/openchat-7b:free", "name": "OpenChat 7B (Free)", "description": "Free chat model", "is_free": True},
            {"id": "deepseek/deepseek-chat-v3-0324:free", "name": "DeepSeek Chat V3 (Free)", "description": "Free efficient model", "is_free": True},
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and efficient", "is_free": False},
            {"id": "openai/gpt-4", "name": "GPT-4", "description": "Most capable model", "is_free": False}
        ]
    
    def chat_completion(self, messages: List[Dict], model: str = "meta-llama/llama-3.2-3b-instruct:free", 
                       temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.__dict__ if hasattr(response, 'usage') else {},
                'model': model,
                'success': True
            }
        except Exception as e:
            return {
                'content': f"Error: {str(e)}",
                'usage': {},
                'model': model,
                'success': False,
                'error': str(e)
            }
    
    def chat_completion_stream(self, messages: List[Dict], model: str = "meta-llama/llama-3.2-3b-instruct:free",
                              temperature: float = 0.7, max_tokens: int = 1000, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in response:
                if chunk and hasattr(chunk, 'choices'):
                    if isinstance(chunk.choices, list) and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and choice.delta:
                            if hasattr(choice.delta, 'content') and choice.delta.content:
                                yield choice.delta.content
                            elif isinstance(choice.delta, dict) and choice.delta.get('content'):
                                yield choice.delta['content']
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def test_connection(self) -> Dict:
        try:
            test_messages = [{"role": "user", "content": "Hi, please respond with 'Connection successful!'"}]
            response = self.chat_completion(
                messages=test_messages,
                model="meta-llama/llama-3.2-3b-instruct:free",
                max_tokens=50
            )
            
            if response['success']:
                return {
                    'success': True,
                    'message': 'Connection successful!',
                    'model_response': response['content']
                }
            else:
                return {
                    'success': False,
                    'message': f"Connection failed: {response.get('error', 'Unknown error')}"
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"Connection test failed: {str(e)}"
            }
