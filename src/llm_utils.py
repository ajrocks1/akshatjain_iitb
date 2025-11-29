import os
import json
import re
import time
from typing import List, Dict, Any, Tuple
from loguru import logger
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Configure API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing from environment variables!")
else:
    genai.configure(api_key=api_key)

_CACHED_MODEL_NAME = None

def get_optimal_model_name() -> str:
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME: return _CACHED_MODEL_NAME

    if os.getenv("GEMINI_MODEL_NAME"):
        return os.getenv("GEMINI_MODEL_NAME")

    try:
        logger.info("Auto-detecting model...")
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        priorities = [
            "models/gemini-2.5-flash", "models/gemini-2.0-flash", 
            "models/gemini-1.5-flash", "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro"
        ]
        
        for p in priorities:
            if p in available:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected: {p}")
                return p
        
        # Fallback
        if available:
            flash_models = [m for m in available if 'flash' in m.lower()]
            _CACHED_MODEL_NAME = flash_models[0] if flash_models else available[0]
            return _CACHED_MODEL_NAME

    except Exception as e:
        logger.error(f"Model detection failed: {e}")
    
    return "models/gemini-1.5-flash"

def build_prompt(text: str) -> str:
    return f"""
    Extract line items from this bill.
    Return STRICT JSON list of objects: [{{ "item_name": "...", "item_amount": ..., "item_rate": ..., "item_quantity": ... }}]
    Use null for missing values. Number fields must be numbers.
    
    BILL TEXT:
    {text}
    """

def clean_json(text: str) -> str:
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def parse_items_with_llm(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
        Tuple containing:
        1. List of parsed items
        2. Dictionary of token usage stats
    """
    model_name = get_optimal_model_name()
    prompt = build_prompt(text)
    
    max_retries = 3
    base_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for attempt in range(max_retries):
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Extract Token Usage
            usage = base_usage.copy()
            if response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }

            parsed = json.loads(clean_json(response.text))
            
            if isinstance(parsed, list): 
                return parsed, usage
            
            return [], usage

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"Quota 429. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"LLM Failed: {e}")
            raise
