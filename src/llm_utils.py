from loguru import logger
import json
import os

# Attempt to import HF Inference client
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

def parse_items_with_llm(text: str) -> list:
    """
    Use an LLM to parse line items from OCR text into structured JSON.
    """
    if not text.strip():
        return []
    logger.info("Invoking LLM to parse items")

    # Define system and user prompts
    system_prompt = {
        "role": "system",
        "content": (
            "You are an assistant that extracts invoice line items from OCR text. "
            "Output a JSON object with key 'items' as a list of objects, each with "
            "item_name, item_quantity, item_rate, and item_amount fields. "
            "If a field is missing, use 0 or an empty value."
        )
    }
    user_prompt = {"role": "user", "content": text}

    llm_output = None

    # Use Hugging Face Inference API if HF_TOKEN is set
    if InferenceClient and os.getenv("HF_TOKEN"):
        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        try:
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo:fastest",  # could use an open model instead
                messages=[system_prompt, user_prompt]
            )
            llm_output = response.choices[0].message.content
            # Log token usage if available
            if hasattr(response, "usage"):
                logger.info(f"LLM token usage: {response.usage}")
            logger.debug(f"LLM raw output: {llm_output}")
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
    else:
        logger.warning("LLM client or HF_TOKEN not configured; skipping LLM parsing")
        return []

    # Parse the JSON output from the LLM
    try:
        result = json.loads(llm_output)
        items = result.get("items", [])
    except Exception as e:
        logger.error(f"Failed to parse JSON from LLM output: {e}")
        items = []
    return items
