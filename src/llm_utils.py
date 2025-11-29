import os
import json
import re
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

# Cache the working model name so we don't query the API every time
_CACHED_MODEL_NAME = None

def get_optimal_model_name() -> str:
    """
    Automatically finds a model that the current API key has access to.
    Prioritizes Gemini 1.5 Flash -> 1.5 Pro -> 1.0 Pro.
    """
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME:
        return _CACHED_MODEL_NAME

    # 1. Try environment variable override first
    env_model = os.getenv("GEMINI_MODEL_NAME")
    if env_model:
        _CACHED_MODEL_NAME = env_model
        return env_model

    logger.info("Auto-detecting best available Gemini model for this API key...")
    
    try:
        # Get all models that support content generation
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        logger.info(f"Available models found: {available_models}")

        # Preference list (Best to Good)
        # Note: list_models() returns names with 'models/' prefix
        priorities = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-pro-001",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]

        # Pick the first priority that exists in the available list
        for p in priorities:
            if p in available_models:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected optimal model: {p}")
                return p
        
        # Fallback: Just take the first available one if none of our preferences match
        if available_models:
            _CACHED_MODEL_NAME = available_models[0]
            logger.warning(f"No preferred model found. Defaulting to: {_CACHED_MODEL_NAME}")
            return _CACHED_MODEL_NAME

    except Exception as e:
        logger.error(f"Failed to auto-detect models: {e}")

    # Ultimate fallback (if listing fails entirely)
    fallback = "models/gemini-1.5-flash"
    logger.warning(f"Auto-detection failed. forcing fallback to: {fallback}")
    return fallback

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
    
    try:
        # Dynamically get the working model name
        model_name = get_optimal_model_name()
        
        logger.info(f"Sending prompt to Gemini (Model: {model_name})...")

        model = GenerativeModel(model_name)
        
        # Enforce JSON generation
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        content = response.text.strip()
        logger.debug(f"Raw Gemini response: {content}")
        
        cleaned_content = clean_json_response(content)
        parsed_data = json.loads(cleaned_content)
        
        if isinstance(parsed_data, list):
            logger.info(f"LLM parsed {len(parsed_data)} items.")
            return parsed_data
        else:
            logger.warning("Model returned JSON, but it was not a list.")
            return []
            
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise
