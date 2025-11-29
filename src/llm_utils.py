import os
import json
import re
from typing import List, Dict, Any
from loguru import logger
import google.generativeai as genai
from google.generativeai import GenerativeModel

# 1. Configure API Key
# Ensure this environment variable is set in Docker or your OS
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing from environment variables!")
else:
    genai.configure(api_key=api_key)

# 2. Configurable Model Name
# We use the pinned version 'gemini-1.5-flash-001' which is more stable than the alias
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-001")

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
    """Removes Markdown code blocks (e.g., ```json) to ensure valid JSON."""
    # Remove ```json or ``` at the start
    text = re.sub(r'^```(json)?', '', raw_text.strip(), flags=re.MULTILINE)
    # Remove ``` at the end
    text = re.sub(r'```$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()

def list_available_models():
    """Helper to debug which models are actually available to this API key."""
    try:
        logger.info("--- DIAGNOSTIC: AVAILABLE MODELS ---")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"Model: {m.name}")
        logger.info("------------------------------------")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")

def parse_items_with_llm(text: str) -> List[Dict[str, Any]]:
    prompt = build_prompt(text)
    
    try:
        logger.info(f"Sending prompt to Gemini (Model: {MODEL_NAME})...")

        model = GenerativeModel(MODEL_NAME)
        
        # Enforce JSON generation for reliability
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        content = response.text.strip()
        logger.debug(f"Raw Gemini response: {content}")
        
        # Clean formatting artifacts
        cleaned_content = clean_json_response(content)

        # Securely parse JSON
        parsed_data = json.loads(cleaned_content)
        
        if isinstance(parsed_data, list):
            logger.info(f"LLM parsed {len(parsed_data)} items.")
            return parsed_data
        else:
            logger.warning("Model returned JSON, but it was not a list.")
            return []
            
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        
        # Auto-Diagnostic: If model not found, list what IS available
        if "404" in str(e) or "not found" in str(e).lower():
            logger.warning(f"Model '{MODEL_NAME}' failed. Checking available models...")
            list_available_models()
            
        # Re-raise so the pipeline knows extraction failed
        raise
