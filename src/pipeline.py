import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc

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

    results = []
    total_items = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        if ext == '.pdf':
            # --- MEMORY OPTIMIZED PDF PROCESSING ---
            try:
                # 1. Get total page count without loading images
                info = pdfinfo_from_path(local_path)
                max_pages = info["Pages"]
                logger.info(f"PDF has {max_pages} pages. Processing sequentially...")

                # 2. Process one page at a time
                for i in range(1, max_pages + 1):
                    logger.info(f"Processing page {i}/{max_pages}...")
                    
                    # Convert ONLY the current page
                    # thread_count=1 reduces CPU load usage
                    page_images = convert_from_path(
                        local_path, 
                        first_page=i, 
                        last_page=i, 
                        fmt='png',
                        thread_count=1
                    )
                    
                    if not page_images:
                        continue
                        
                    pil_image = page_images[0]
                    
                    # Save to temp file for Vision Model
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                        pil_image.save(tmp_img.name, format='PNG')
                        temp_img_path = tmp_img.name

                    # Process with Gemini
                    try:
                        items, usage = parse_items_with_llm(temp_img_path)
                    except Exception as e:
                        logger.warning(f"Vision LLM failed on page {i}: {e}")
                        items = []
                        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    
                    # Cleanup immediately
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    
                    # Explicitly free memory
                    del pil_image
                    del page_images
                    gc.collect()

                    # Aggregate Results
                    for item in items:
                        item["item_name"] = str(item.get("item_name", "")).strip()
                        for field in ["item_quantity", "item_rate", "item_amount"]:
                            try:
                                val = item.get(field)
                                item[field] = float(val) if val is not None else 0.0
                            except:
                                item[field] = 0.0

                    total_items += len(items)
                    for k in token_usage:
                        token_usage[k] += usage.get(k, 0)

                    results.append({
                        "page_no": str(i),
                        "page_type": "Bill Detail",
                        "bill_items": items
                    })

            except Exception as e:
                 logger.error(f"PDF processing failed: {e}")
                 raise e

        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            # Single Image Processing
            logger.info("Processing single image...")
            try:
                items, usage = parse_items_with_llm(local_path)
            except Exception as e:
                logger.warning(f"Vision LLM failed: {e}")
                items = []
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            for item in items:
                item["item_name"] = str(item.get("item_name", "")).strip()
                for field in ["item_quantity", "item_rate", "item_amount"]:
                    try:
                        val = item.get(field)
                        item[field] = float(val) if val is not None else 0.0
                    except:
                        item[field] = 0.0

            total_items += len(items)
            for k in token_usage:
                token_usage[k] += usage.get(k, 0)

            results.append({
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": items
            })
        
        else:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")

    finally:
        # Final Cleanup of the downloaded file
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
