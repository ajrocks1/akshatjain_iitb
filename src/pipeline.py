import os
import tempfile
import requests
from pdf2image import convert_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger

def download_url_to_file(url):
    """Downloads file and preserves extension."""
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    path_no_query = url.split('?')[0]
    suffix = os.path.splitext(path_no_query)[1] or ".bin"
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.flush(); tmp.close()
    return tmp.name

def process_bill(file_url: str) -> dict:
    logger.info(f"Starting bill processing for URL: {file_url}")
    local_path = download_url_to_file(file_url)
    ext = os.path.splitext(local_path)[1].lower()

    # 1. Convert to List of Image Paths
    if ext == '.pdf':
        try:
            images = convert_from_path(local_path, fmt='png')
            pages = []
            for i, img in enumerate(images, start=1):
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                img.save(temp_img.name, format='PNG')
                pages.append(temp_img.name)
        except Exception as e:
             logger.error(f"PDF conversion failed: {e}")
             raise e
    elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
        pages = [local_path]
    else:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

    results = []
    total_items = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # 2. Process Each Page
    for idx, image_path in enumerate(pages):
        logger.info(f"Processing page {idx+1}...")
        try:
            # Assume Bill Detail since we skipped OCR
            page_type = "Bill Detail"

            try:
                items, usage = parse_items_with_llm(image_path)
            except Exception as e:
                logger.warning(f"Vision LLM failed on page {idx+1}: {e}")
                items = []
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            # Normalize Data Types
            for item in items:
                item["item_name"] = str(item.get("item_name", "")).strip()
                for field in ["item_quantity", "item_rate", "item_amount"]:
                    try:
                        val = item.get(field)
                        item[field] = float(val) if val is not None else 0.0
                    except:
                        item[field] = 0.0

            total_items += len(items)
            
            # Aggregate Usage
            for k in token_usage:
                token_usage[k] += usage.get(k, 0)

            results.append({
                "page_no": str(idx + 1),
                "page_type": page_type,
                "bill_items": items
            })

        except Exception as e:
            logger.error(f"Error processing page {idx+1}: {e}")
        finally:
            if image_path != local_path and os.path.exists(image_path):
                os.remove(image_path)

    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except Exception as cleanup_error:
        logger.warning(f"Cleanup failed: {cleanup_error}")

    # --- EXACT REQUIRED OUTPUT STRUCTURE ---
    return {
        "is_success": True,
        "token_usage": token_usage,  # <--- NOW AT ROOT LEVEL
        "data": {
            "pagewise_line_items": results,
            "total_item_count": total_items
        }
    }
