import os
import json
import re
import time
import PIL.Image
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
    """
    Finds the best Vision-capable model. 
    Prioritizes Flash 1.5/2.0 for speed and multimodal capabilities.
    """
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME: return _CACHED_MODEL_NAME
    
    # Check env override
    if os.getenv("GEMINI_MODEL_NAME"):
        return os.getenv("GEMINI_MODEL_NAME")

    try:
        logger.info("Auto-detecting Vision model...")
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority list for Vision tasks
        priorities = [
            "models/gemini-2.5-flash", 
            "models/gemini-2.0-flash", 
            "models/gemini-1.5-flash", 
            "models/gemini-1.5-pro"
        ]
        
        for p in priorities:
            if p in available:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected Vision Model: {p}")
                return p
        
        # Fallback
        _CACHED_MODEL_NAME = "models/gemini-1.5-flash"
        return _CACHED_MODEL_NAME

    except Exception as e:
        logger.error(f"Model detection failed: {e}")
        return "models/gemini-1.5-flash"

def clean_json(text: str) -> str:
    """Removes Markdown formatting to ensure valid JSON."""
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def parse_items_with_llm(image_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Sends the IMAGE directly to Gemini Vision.
    Returns: (items_list, token_usage_dict)
    """
    model_name = get_optimal_model_name()
    
    prompt = """
    Analyze this bill image. Extract all line items as a strictly formatted JSON list.
    
    CRITICAL INSTRUCTIONS:
    1. Read item names exactly as they appear, but correct obvious typos (e.g., 'Consutaion' -> 'Consultation').
    2. Contextualize codes: 'S6204' is likely 'SG204' if other items use 'SG'. Fix these OCR-like errors.
    3. Return fields: item_name (string), item_amount (number), item_rate (number), item_quantity (number).
    4. Use null if a field is missing.
    5. Ignore page headers/footers.
    """
    
    base_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        img = PIL.Image.open(image_path)
    except Exception as e:
        logger.error(f"Could not load image for Vision: {e}")
        return [], base_usage

    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = GenerativeModel(model_name)
            
            # Send [Prompt, Image] - This is the Multimodal call
            response = model.generate_content(
                [prompt, img], 
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
            
            logger.warning("Model output was not a list.")
            return [], usage

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"Quota 429. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"Vision LLM Failed: {e}")
            raise
