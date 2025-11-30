import os
import json
import re
import time
import random
import PIL.Image
import asyncio
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
        
        # PRIORITIES BASED ON YOUR RATE LIMIT TABLE
        # 1. Gemini 2.0 Flash: 2,000 RPM (Best Balance of Speed/Quality)
        # 2. Gemini 2.5 Flash-Lite: 4,000 RPM (Max Speed Backup)
        # 3. Gemini 2.5 Flash: 1,000 RPM (Quality Backup)
        priorities = [
            "models/gemini-2.0-flash-001",
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash-lite-preview",
            "models/gemini-2.5-flash-lite",
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash"
        ]
        
        for p in priorities:
            if p in available:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected High-Speed Model: {p}")
                return p
        
        # Intelligent Fallback: Look for any "2.0-flash" or "flash-lite"
        for m in available:
            if "2.0-flash" in m and "exp" not in m:
                _CACHED_MODEL_NAME = m
                return m
            if "flash-lite" in m:
                _CACHED_MODEL_NAME = m
                return m

        _CACHED_MODEL_NAME = "models/gemini-1.5-flash"
        return _CACHED_MODEL_NAME

    except Exception:
        return "models/gemini-1.5-flash"

def clean_json(text: str) -> str:
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()

async def parse_items_with_llm(image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    model_name = get_optimal_model_name()
    
    # --- HYPER-TUNED PROMPT FOR ACCURACY & SPEED ---
    prompt = """
    Analyze this medical bill.
    
    ### 1. CLASSIFY PAGE TYPE (Strict Visual Rules)
    - "Pharmacy": Contains medicines, batch #, expiry, "Sch", "Mfg".
    - "Final Bill": Summary of charges, "Room Rent", "Grand Total", "Net Amt".
    - "Bill Detail": Line-item breakdown of services/tests, "Test Particulars".

    ### 2. EXTRACT ITEMS
    - Fields: item_name, item_amount, item_rate, item_quantity.
    - Rules: Use null for missing data. Fix typos.
    - FLATTEN: If multiple bills appear on one page, merge ALL items into a single list. Do not nest.

    RETURN STRICT JSON:
    { "page_type": "...", "items": [ ... ] }
    """
    
    base_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        # Offload image loading to thread to prevent blocking async loop
        img = await asyncio.to_thread(PIL.Image.open, image_path)
    except Exception as e:
        logger.error(f"Could not load image: {e}")
        return "Bill Detail", [], base_usage

    max_retries = 5
    for attempt in range(max_retries):
        try:
            model = GenerativeModel(model_name)
            
            # Async Generation
            response = await model.generate_content_async(
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
            
            # --- FLATTENING LOGIC ---
            raw_items = []
            p_type = "Bill Detail"

            if isinstance(parsed, list):
                raw_items = parsed
            else:
                p_type = parsed.get("page_type", "Bill Detail")
                raw_items = parsed.get("items", [])

            flat_items = []
            for item in raw_items:
                if isinstance(item, dict) and "items" in item and isinstance(item["items"], list):
                    flat_items.extend(item["items"])
                else:
                    flat_items.append(item)
            
            return p_type, flat_items, usage

        except Exception as e:
            error_str = str(e)
            
            if "404" in error_str:
                logger.warning(f"Model {model_name} 404. Resetting cache...")
                global _CACHED_MODEL_NAME
                _CACHED_MODEL_NAME = None 
                continue

            if "429" in error_str and attempt < max_retries - 1:
                # With 2000 RPM, we can be aggressive. Short sleep.
                sleep_time = (1.5 * (attempt + 1)) + random.uniform(0.1, 0.5)
                logger.warning(f"Quota 429. Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                continue
                
            logger.error(f"Vision LLM Failed: {e}")
            raise
    
    return "Bill Detail", [], base_usage
