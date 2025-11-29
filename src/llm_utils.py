import os
import json
import re
import time
import random
import PIL.Image
from typing import List, Dict, Any, Tuple
from loguru import logger
import google.generativeai as genai
from google.generativeai import GenerativeModel

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing!")
else:
    genai.configure(api_key=api_key)

_CACHED_MODEL_NAME = None

def get_optimal_model_name() -> str:
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME: return _CACHED_MODEL_NAME
    
    if os.getenv("GEMINI_MODEL_NAME"):
        return os.getenv("GEMINI_MODEL_NAME")

    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # PRIORITIZE PINNED VERSIONS (Most Stable)
        priorities = [
            "models/gemini-1.5-flash-001",  # STABLE PINNED (Best for Prod)
            "models/gemini-1.5-flash-002",  # NEWER STABLE
            "models/gemini-1.5-flash",      # ALIAS (Can be flaky)
            "models/gemini-1.5-flash-8b",
            "models/gemini-1.5-pro-001",
        ]
        
        for p in priorities:
            if p in available:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected High-Speed Model: {p}")
                return p
        
        # Fallback to the most reliable pinned version
        _CACHED_MODEL_NAME = "models/gemini-1.5-flash-001"
        return _CACHED_MODEL_NAME

    except Exception:
        # Fallback if list_models fails
        return "models/gemini-1.5-flash-001"

def clean_json(text: str) -> str:
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def parse_items_with_llm(image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    model_name = get_optimal_model_name()
    
    prompt = """
    Analyze this medical bill image.
    
    1. CLASSIFY the page into exactly one of these types:
       - "Pharmacy": If it lists medicines/drugs with batch numbers.
       - "Final Bill": If it shows summary totals like 'Room Charges', 'Grand Total'.
       - "Bill Detail": For standard line-item breakdowns.
    
    2. EXTRACT all line items.
       - Fields: item_name, item_amount, item_rate, item_quantity.
       - Use null if missing.
       - Correct typos.

    RETURN STRICT JSON:
    {
        "page_type": "...",
        "items": [ ... ]
    }
    """
    
    base_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        img = PIL.Image.open(image_path)
    except Exception as e:
        logger.error(f"Could not load image: {e}")
        return "Bill Detail", [], base_usage

    max_retries = 5 
    for attempt in range(max_retries):
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, img], 
                generation_config={"response_mime_type": "application/json"}
            )
            
            usage = base_usage.copy()
            if response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }

            parsed = json.loads(clean_json(response.text))
            
            if isinstance(parsed, list):
                return "Bill Detail", parsed, usage
            
            p_type = parsed.get("page_type", "Bill Detail")
            items = parsed.get("items", [])
            
            return p_type, items, usage

        except Exception as e:
            error_str = str(e)
            
            # HANDLE 404 MODEL NOT FOUND (Dynamic Switch)
            if "404" in error_str and "models/" in error_str:
                logger.warning(f"Model {model_name} not found (404). Switching to fallback...")
                # Switch globally to the generic alias if specific pin fails, or vice versa
                global _CACHED_MODEL_NAME
                if "001" in model_name:
                    _CACHED_MODEL_NAME = "models/gemini-1.5-flash"
                else:
                    _CACHED_MODEL_NAME = "models/gemini-1.5-flash-001"
                model_name = _CACHED_MODEL_NAME
                continue

            if "429" in error_str and attempt < max_retries - 1:
                sleep_time = (2 * (attempt + 1)) + random.uniform(0.1, 2.0)
                logger.warning(f"Quota 429. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                continue
                
            logger.error(f"Vision LLM Failed: {e}")
            raise
