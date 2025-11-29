from pdf2image import convert_from_path
from ocr import extract_text_from_image
from page_classifier import classify_page
from llm_utils import parse_items_with_llm
from loguru import logger
import os
import tempfile

def process_bill(file_path: str) -> dict:
    """
    Main pipeline to process a bill PDF/image.
    Steps: convert PDF->images, OCR text, parse items, classify pages, and format response.
    """
    logger.info(f"Starting bill processing for file: {file_path}")

    # Prepare list of page image files
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        try:
            logger.info("Converting PDF to images...")
            images = convert_from_path(file_path, fmt='png')
            logger.info(f"Converted PDF to {len(images)} images")
            pages = []
            # Save each page image to a temp file
            for i, img in enumerate(images, start=1):
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                img.save(temp_file.name, format='PNG')
                pages.append(temp_file.name)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
    elif ext in ['.png', '.jpg', '.jpeg']:
        pages = [file_path]
    else:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}")

    page_results = []
    total_items = 0

    # Process each page image
    for page_no, page_file in enumerate(pages, start=1):
        logger.info(f"OCR on page {page_no}...")
        try:
            text = extract_text_from_image(page_file)
        except Exception as e:
            logger.error(f"OCR failed on page {page_no}: {e}")
            text = ""

        # Classify page type
        page_type = classify_page(text)
        logger.info(f"Page {page_no} classified as {page_type}")

        # Parse line items using LLM
        try:
            items = parse_items_with_llm(text)
            logger.info(f"Extracted {len(items)} items from page {page_no}")
        except Exception as e:
            logger.error(f"Item parsing failed on page {page_no}: {e}")
            items = []

        # Ensure each item has all required fields (default 0.0 if missing)
        for item in items:
            item.setdefault("item_quantity", 0.0)
            item.setdefault("item_rate", 0.0)
            item.setdefault("item_amount", 0.0)
            item.setdefault("item_name", item.get("item_name", ""))

            # Convert to float if possible, else default to 0.0
            try:
                item["item_quantity"] = float(item["item_quantity"]) if item["item_quantity"] else 0.0
            except:
                item["item_quantity"] = 0.0
            try:
                item["item_rate"] = float(item["item_rate"]) if item["item_rate"] else 0.0
            except:
                item["item_rate"] = 0.0
            try:
                item["item_amount"] = float(item["item_amount"]) if item["item_amount"] else 0.0
            except:
                item["item_amount"] = 0.0

        page_results.append({
            "page_no": str(page_no),
            "page_type": page_type,
            "bill_items": items
        })
        total_items += len(items)

    # Clean up any temporary image files
    for page_file in pages:
        try:
            if os.path.exists(page_file):
                os.remove(page_file)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {page_file}: {e}")

    response = {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": page_results,
            "total_item_count": total_items
        }
    }
    logger.info("Completed bill processing")
    return response
