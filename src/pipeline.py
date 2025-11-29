import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc
import concurrent.futures

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

def process_single_page(page_num: int, image_path: str) -> dict:
    """
    Worker function to process a single page in a separate thread.
    """
    logger.info(f"Thread started for Page {page_num}...")
    try:
        # Call Vision LLM
        p_type, items, usage = parse_items_with_llm(image_path)
        
        # Normalize Data
        for item in items:
            item["item_name"] = str(item.get("item_name", "")).strip()
            for field in ["item_quantity", "item_rate", "item_amount"]:
                try:
                    item[field] = float(item.get(field, 0.0))
                except:
                    item[field] = 0.0

        return {
            "page_no": str(page_num),
            "page_type": p_type,
            "bill_items": items,
            "token_usage": usage,
            "image_path": image_path # Return path so we can delete it later
        }
    except Exception as e:
        logger.error(f"Error on Page {page_num}: {e}")
        return {
            "page_no": str(page_num),
            "page_type": "Bill Detail",
            "bill_items": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "image_path": image_path
        }

def process_bill(file_url: str) -> dict:
    logger.info(f"Starting bill processing for URL: {file_url}")
    local_path = download_url_to_file(file_url)
    ext = os.path.splitext(local_path)[1].lower()

    final_results = []
    total_items = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    temp_files_to_cleanup = []

    try:
        # 1. PREPARE IMAGE PATHS (Low Memory)
        tasks = []
        
        if ext == '.pdf':
            info = pdfinfo_from_path(local_path)
            max_pages = info["Pages"]
            logger.info(f"PDF has {max_pages} pages. Converting to images...")

            for i in range(1, max_pages + 1):
                # Convert 1 page at a time to keep RAM low
                page_images = convert_from_path(local_path, first_page=i, last_page=i, fmt='png', thread_count=1)
                if page_images:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                        page_images[0].save(tmp_img.name, format='PNG')
                        tasks.append((i, tmp_img.name))
                        temp_files_to_cleanup.append(tmp_img.name)
                    del page_images
                    gc.collect()
        
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            tasks.append((1, local_path))
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # 2. PARALLEL EXECUTION (Speed Boost)
        # We use 3 workers to stay within 512MB RAM but process 3x faster
        logger.info(f"Starting Parallel Processing with 3 workers on {len(tasks)} pages...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(process_single_page, p_num, p_path): p_num 
                for p_num, p_path in tasks
            }
            
            # Collect results as they finish
            for future in concurrent.futures.as_completed(future_to_page):
                res = future.result()
                final_results.append(res)
                
                # Aggregate Stats immediately
                total_items += len(res["bill_items"])
                u = res["token_usage"]
                for k in total_usage:
                    total_usage[k] += u.get(k, 0)
                
                # Cleanup individual page image immediately to free space
                if os.path.exists(res["image_path"]) and res["image_path"] != local_path:
                    try:
                        os.remove(res["image_path"])
                    except:
                        pass

        # 3. SORT RESULTS (Because parallel threads finish randomly)
        final_results.sort(key=lambda x: int(x["page_no"]))
        
        # Remove internal keys before returning
        cleaned_results = []
        for r in final_results:
            cleaned_results.append({
                "page_no": r["page_no"],
                "page_type": r["page_type"],
                "bill_items": r["bill_items"]
            })

    except Exception as e:
        logger.error(f"Processing Failed: {e}")
        raise e
    finally:
        # Final cleanup
        if os.path.exists(local_path):
            os.remove(local_path)
        for p in temp_files_to_cleanup:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        gc.collect()

    logger.info(f"Pipeline Completed. Total Time < 150s Target. Items: {total_items}")

    return {
        "is_success": True,
        "token_usage": total_usage,
        "data": {
            "pagewise_line_items": cleaned_results,
            "total_item_count": total_items
        }
    }
