# src/llm_utils.py
import os
from loguru import logger
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def build_prompt(text: str) -> str:
    return f"""
Extract line items from the following bill text.
Each item should contain item_name, item_amount, item_rate, and item_quantity.

Return as a list of dictionaries like:
[{{"item_name": "...", "item_amount": ..., "item_rate": ..., "item_quantity": ...}}, ...]

BILL TEXT:
{text}
"""

def parse_items_with_llm(text: str):
    prompt = build_prompt(text)
    try:
        logger.info("Sending prompt to Gemini...")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        if response.text:
            # Evaluate Gemini's response into Python list
            parsed = eval(response.text.strip())
            logger.info(f"LLM parsed {len(parsed)} items.")
            return parsed
        else:
            raise ValueError("Empty response from Gemini.")
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise
