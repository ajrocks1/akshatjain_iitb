# src/llm_utils.py
import os
from loguru import logger
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(text: str) -> str:
    return f"""
Extract line items from the following bill text.
Each item should contain item_name, item_amount, item_rate, and item_quantity.

Return the result as a Python list of dictionaries in this format:
[{{"item_name": "...", "item_amount": ..., "item_rate": ..., "item_quantity": ...}}, ...]

BILL TEXT:
{text}
"""

def parse_items_with_llm(text: str):
    prompt = build_prompt(text)
    try:
        logger.info("Sending prompt to Gemini...")

        model = GenerativeModel("gemini-1.5-flash")  # or "gemini-pro" if needed
        response = model.generate_content(prompt)

        content = response.text.strip()
        logger.debug(f"Raw Gemini response: {content}")
        
        parsed = eval(content)  # Optionally: use `json.loads()` if response is JSON
        logger.info(f"LLM parsed {len(parsed)} items.")
        return parsed
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise
