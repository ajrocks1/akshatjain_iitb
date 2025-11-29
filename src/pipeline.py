import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc
import concurrent.futures
import time

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

def process_page_task(page_num: int, file_path: str, is_pdf: bool) -> dict:
    """
    Worker function: Converts (if PDF) AND Extracts in one go.
    """
    start_time = time.time()
    temp_img_path = None
    
    try:
        # 1. CONVERT ON DEMAND (Lazy Loading)
        if is_pdf:
            # logger.info(f"P{page_num}: Converting...")
            images = convert_from_path(
                file_path, 
                first_page=page_num, 
                last_page=page_num, 
                fmt='png', 
                thread_count=1 # Low CPU per thread
            )
            if not images:
                raise ValueError("PDF Page conversion returned no images")
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                images[0].save(tmp.name, format='PNG')
                temp_img_path = tmp.name
            
            # Free RAM immediately
            del images
            gc.collect()
        else:
            # It's already an image
            temp_img_path = file_path

        # 2. EXTRACT WITH AI
        # logger.info(f"P{page_num}: Analyzing...")
        p_type, items, usage = parse_items_with_llm(temp_img_path)
        
        duration = time.time() - start_time
        logger.info(f"Page {page_num} Done in {duration:.1f}s. Items: {len(items)}")

        # 3. NORMALIZE
        for item in items:
            item["item_name"] = str(item.get("item_name", "")).strip()
            for field in ["item_quantity", "item_rate", "item_amount"]:
                try:
                    val = item.get(field)
                    item[field] = float(val) if val is not None else 0.0
                except:
                    item[field] = 0.0

        return {
            "page_no": str(page_num),
            "page_type": p_type,
            "bill_items": items,
            "token_usage": usage,
            "temp_path": temp_img_path if is_pdf else None
        }

    except Exception as e:
        logger.error(f"Error on Page {page_num}: {e}")
        return {
            "page_no": str(page_num),
            "page_type": "Bill Detail",
            "bill_items": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "temp_path": temp_img_path
        }

def process_bill(file_url: str) -> dict:
    start_run = time.time()
    logger.info(f"Starting bill processing for URL: {file_url}")
    local_path = download_url_to_file(file_url)
    ext = os.path.splitext(local_path)[1].lower()

    final_results = []
    total_items = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # Track temp files to cleanup
    cleanup_list = [local_path]

    try:
        tasks = []
        is_pdf = False

        # 1. SETUP TASKS (Instant)
        if ext == '.pdf':
            is_pdf = True
            info = pdfinfo_from_path(local_path)
            max_pages = info["Pages"]
            logger.info(f"PDF has {max_pages} pages. Launching parallel workers immediately...")
            
            for i in range(1, max_pages + 1):
                tasks.append(i)
        
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            logger.info("Processing single image...")
            tasks.append(1)
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # 2. PARALLEL EXECUTION (Conversion + Extraction)
        # 3 Workers is the sweet spot for Render Free Tier (RAM safe)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_page = {
                executor.submit(process_page_task, p_num, local_path, is_pdf): p_num 
                for p_num in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                res = future.result()
                final_results.append(res)
                
                # Aggregate
                total_items += len(res["bill_items"])
                u = res["token_usage"]
                for k in total_usage:
                    total_usage[k] += u.get(k, 0)
                
                # Cleanup Individual Page Image immediately
                if res["temp_path"] and os.path.exists(res["temp_path"]):
                    try:
                        os.remove(res["temp_path"])
                    except:
                        pass

        # 3. FINALIZE
        final_results.sort(key=lambda x: int(x["page_no"]))
        
        cleaned_results = []
        for r in final_results:
            cleaned_results.append({
                "page_no": r["page_no"],
                "page_type": r["page_type"],
                "bill_items": r["bill_items"]
            })

    finally:
        # Global Cleanup
        for f in cleanup_list:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        gc.collect()

    total_time = time.time() - start_run
    logger.info(f"RUN COMPLETE. Time: {total_time:.2f}s (Target <150s). Items: {total_items}")

    return {
        "is_success": True,
        "token_usage": total_usage,
        "data": {
            "pagewise_line_items": cleaned_results,
            "total_item_count": total_items
        }
    }
