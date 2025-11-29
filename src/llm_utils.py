import os
import json
import re
import time
from typing import List, Dict, Any
from loguru import logger
import google.generativeai as genai
from google.generativeai import GenerativeModel

# 1. Configure API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing from environment variables!")
else:
    genai.configure(api_key=api_key)

# Cache the working model name
_CACHED_MODEL_NAME = None

def get_optimal_model_name() -> str:
    """
    Finds the best available model. 
    Prioritizes 'Flash' models (high speed/quota) over 'Pro' or 'Preview' models.
    """
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME:
        return _CACHED_MODEL_NAME

    # Allow manual override
    env_model = os.getenv("GEMINI_MODEL_NAME")
    if env_model:
        _CACHED_MODEL_NAME = env_model
        return env_model

    logger.info("Auto-detecting best available Gemini model...")
    
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Updated Priority List based on your logs (Newer models first)
        # We prioritize FLASH models because they have higher rate limits (RPM)
        priorities = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-flash-8b",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro"
        ]

        # check for matches
        for p in priorities:
            if p in available_models:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected optimal high-quota model: {p}")
                return p
        
        # Smart Fallback: If no exact match, look for ANY 'flash' model
        for m in available_models:
            if "flash" in m.lower():
                _CACHED_MODEL_NAME = m
                logger.warning(f"No preferred exact match. Falling back to generic flash model: {m}")
                return m

        # Ultimate Fallback
        if available_models:
            _CACHED_MODEL_NAME = available_models[0]
            logger.warning(f"No Flash model found. Defaulting to: {_CACHED_MODEL_NAME}")
            return _CACHED_MODEL_NAME

    except Exception as e:
        logger.error(f"Model auto-detection failed: {e}")

    return "models/gemini-1.5-flash"

def build_prompt(text: str) -> str:
    return f"""
    You are an expert data extraction assistant. 
    Extract line items from the provided bill text.

    REQUIREMENTS:
    1. Extract: item_name, item_amount, item_rate, and item_quantity.
    2. If a value is missing, use null.
    3. Ensure numerical values are numbers, not strings (e.g., 10.5 not "10.5").
    
    Output must be a strictly valid JSON array of objects.

    BILL TEXT:
    {text}
    """

def clean_json_response(raw_text: str) -> str:
    text = re.sub(r'^```(json)?', '', raw_text.strip(), flags=re.MULTILINE)
    text = re.sub(r'```$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()

def parse_items_with_llm(text: str) -> List[Dict[str, Any]]:
    prompt = build_prompt(text)
    model_name = get_optimal_model_name()
    
    # Retry configuration
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Sending prompt to Gemini (Model: {model_name}, Attempt: {attempt+1})...")
            model = GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            content = response.text.strip()
            cleaned_content = clean_json_response(content)
            parsed_data = json.loads(cleaned_content)
            
            if isinstance(parsed_data, list):
                logger.info(f"LLM parsed {len(parsed_data)} items.")
                return parsed_data
            else:
                logger.warning("Model returned JSON, but it was not a list.")
                return []

        except Exception as e:
            error_str = str(e)
            
            # Check for Quota (429) or Overloaded (503) errors
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    logger.warning(f"Quota exceeded (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            logger.error(f"Gemini call failed: {e}")
            raise
