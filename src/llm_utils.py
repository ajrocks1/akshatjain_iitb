# src/llm_utils.py
import os
from openai import OpenAI
from loguru import logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(text: str) -> str:
    return f"""
You are a smart invoice parser.

Below is a scanned bill page's text content.
Your job is to extract the list of line items present on the page.

Each line item should include:
- item_name: the product/service name
- item_quantity: how many units
- item_rate: price per unit
- item_amount: total price for that line item

If quantity, rate, or amount is missing, just return 0.0.

Return only a JSON array of objects with the format:
[
  {{
    "item_name": "Paracetamol",
    "item_quantity": 1.0,
    "item_rate": 35.0,
    "item_amount": 35.0
  }},
  ...
]

Here is the extracted text from the bill page:
--------------------
{text}
--------------------
"""

def parse_items_with_llm(text: str):
    logger.info("Sending prompt to OpenAI GPT...")
    prompt = build_prompt(text)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract line items from bills into JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        reply = response.choices[0].message.content
        logger.debug(f"LLM raw response: {reply}")
        import json
        return json.loads(reply)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []
