# src/llm_utils.py
import openai
import os
import json
from loguru import logger
from dotenv import load_dotenv; load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_items_with_llm(text: str):
    prompt = f"""
You are a helpful bill parser. Extract line items from the following bill text.
Each item should have:
- item_name (string)
- item_quantity (float)
- item_rate (float)
- item_amount (float)

Return a JSON list of such items only.

{text}
"""

    try:
        logger.info("Sending prompt to OpenAI GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an invoice parsing expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        reply = response.choices[0].message.content.strip()
        usage = response.usage
        logger.info(f"OpenAI responded. Tokens used: {usage}")

        try:
            items = json.loads(reply)
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON, returning empty item list")
            items = []

        return items, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
