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
        
        # STRATEGY: Accuracy First
        priorities = [
            "models/gemini-2.0-flash-001",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-pro", 
            "models/gemini-2.5-flash",
        ]
        
        for p in priorities:
            if p in available:
                _CACHED_MODEL_NAME = p
                logger.info(f"Selected High-Accuracy Model: {p}")
                return p
        
        for m in available:
            if "2.0-flash" in m and "exp" not in m:
                _CACHED_MODEL_NAME = m
                return m

        return "models/gemini-1.5-flash"

    except Exception:
        return "models/gemini-1.5-flash"

def clean_json(text: str) -> str:
    return re.sub(r'^```(json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def parse_items_with_llm(image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    model_name = get_optimal_model_name()
    
    # --- FINAL CORRECTED PROMPT ---
    prompt = """
    Act as an Expert Pharmacist and Handwriting Analyst. Analyze this medical bill image.

    ### 1. ROBUSTNESS RULES
    - The image may be HANDWRITTEN and messy. Use context to identify medicine names (e.g. 'Pantaviz', 'Divalgress', 'Augmentin').
    - If the image contains TWO separate receipts (left and right), EXTRACT items from BOTH.
    - IGNORE purely summary rows like "Total", "Grand Total", "Balance", "Round Off".

    ### 2. CLASSIFY THE PAGE TYPE (Pick exactly one):
       - "Pharmacy": 
         * Keywords: "Pharmacy", "Chemist", "Druggist", "Medical Store".
         * Columns: "Batch", "B.No", "Expiry", "Exp", "Mfg".
       
       - "Final Bill": 
         * Keywords: "Room & Nursing Charges", "Professional Fees", "Bill Summary", "IP Bill", "Advance".
         * Shows CATEGORY TOTALS. Usually has a "Grand Total" at the bottom.
       
       - "Bill Detail": 
         * Detailed breakdowns of SERVICES (e.g. "Urine Routine", "X-Ray Chest").
         * Dates listed per line item.

    ### 3. EXTRACTION RULES (Strict Judge Compliance)
    - Fields: item_name, item_amount, item_rate, item_quantity.
    
    - **QUANTITY HANDLING (Pack vs Total)**:
      - If Quantity is written as **"3 x 10"** or **"2 x 15"**, extract ONLY the **First Number** (the number of packs).
      - Example: "3 x 10" -> Quantity: 3.
      - Example: "2 x 15" -> Quantity: 2.
      - Remove symbols like ')' or '.' (e.g., "10)" -> 10).

    - **NO CATEGORY HEADERS**:
      - **DO NOT extract Section Headers** (e.g., "Consultation: 1950") if they just summarize the items below them.

    - **COMBINE SPLIT ROWS**:
      - If a single row has a Code and a Description, **COMBINE THEM** into one item name.
      
    - **RATE HANDLING**: 
      - If 'Rate' column is missing, empty, or not visible: **RETURN 0**. 
      - **DO NOT CALCULATE IT**.
      
    - **Typo Correction**: Fix spelling errors in medicine names.

    RETURN STRICT JSON:
    {
        "page_type": "Pharmacy | Final Bill | Bill Detail",
        "items": [ 
            { "item_name": "...", "item_quantity": 0.0, "item_rate": 0.0, "item_amount": 0.0 }
        ]
    }
    """
    
    base_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        img = PIL.Image.open(image_path)
    except Exception as e:
        logger.error(f"Could not load image: {e}")
        return "Bill Detail", [], base_usage

    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, img], 
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0 
                }
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
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
            
    return "Bill Detail", [], base_usage
