import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc

def download_url_to_file(url):
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

    results = []
    total_items = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        if ext == '.pdf':
            try:
                info = pdfinfo_from_path(local_path)
                max_pages = info["Pages"]
                logger.info(f"PDF has {max_pages} pages.")

                for i in range(1, max_pages + 1):
                    logger.info(f"Processing page {i}/{max_pages}...")
                    page_images = convert_from_path(local_path, first_page=i, last_page=i, fmt='png', thread_count=1)
                    
                    if not page_images: continue
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                        page_images[0].save(tmp_img.name, format='PNG')
                        temp_img_path = tmp_img.name

                    try:
                        # --- UPDATED UNPACKING ---
                        # Now getting page_type dynamically from Gemini
                        p_type, items, usage = parse_items_with_llm(temp_img_path)
                    except Exception as e:
                        logger.warning(f"Vision LLM failed on page {i}: {e}")
                        p_type = "Bill Detail"
                        items = []
                        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)
                    del page_images
                    gc.collect()

                    for item in items:
                        item["item_name"] = str(item.get("item_name", "")).strip()
                        for field in ["item_quantity", "item_rate", "item_amount"]:
                            try:
                                item[field] = float(item.get(field, 0.0))
                            except:
                                item[field] = 0.0

                    total_items += len(items)
                    for k in token_usage: token_usage[k] += usage.get(k, 0)

                    results.append({
                        "page_no": str(i),
                        "page_type": p_type, # Using the AI detected type
                        "bill_items": items
                    })

            except Exception as e:
                 logger.error(f"PDF processing failed: {e}")
                 raise e

        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            logger.info("Processing single image...")
            try:
                p_type, items, usage = parse_items_with_llm(local_path)
            except Exception as e:
                logger.warning(f"Vision LLM failed: {e}")
                p_type = "Bill Detail"
                items = []
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            for item in items:
                item["item_name"] = str(item.get("item_name", "")).strip()
                for field in ["item_quantity", "item_rate", "item_amount"]:
                    try:
                        item[field] = float(item.get(field, 0.0))
                    except:
                        item[field] = 0.0

            total_items += len(items)
            for k in token_usage: token_usage[k] += usage.get(k, 0)

            results.append({
                "page_no": "1",
                "page_type": p_type,
                "bill_items": items
            })
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            gc.collect()

    return {
        "is_success": True,
        "token_usage": token_usage,
        "data": {
            "pagewise_line_items": results,
            "total_item_count": total_items
        }
    }
