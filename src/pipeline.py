import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc
import concurrent.futures
import time
import PIL.Image

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

def optimize_image(image_path_or_obj, is_obj=False):
    """
    Resizes image to max 1024px and saves as optimized JPEG.
    Returns path to temp JPEG.
    """
    try:
        if is_obj:
            img = image_path_or_obj
        else:
            img = PIL.Image.open(image_path_or_obj)
            
        # Resize to speed up processing (Max 1024x1024)
        img.thumbnail((1024, 1024))
        
        # Convert to RGB (in case of RGBA png)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Save as compressed JPEG
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name, format='JPEG', quality=85)
            return tmp.name
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return None

def process_page_task(page_num: int, file_path: str, is_pdf: bool) -> dict:
    """
    Worker: Converts -> Resizes -> Extracts.
    """
    start_time = time.time()
    temp_jpeg_path = None
    
    try:
        # 1. GET IMAGE
        if is_pdf:
            # Convert PDF Page
            images = convert_from_path(
                file_path, 
                first_page=page_num, 
                last_page=page_num, 
                fmt='jpeg', 
                thread_count=1
            )
            if not images:
                raise ValueError("PDF Page conversion returned no images")
            
            # Optimize immediately
            temp_jpeg_path = optimize_image(images[0], is_obj=True)
            
            # Free RAM
            del images
            gc.collect()
        else:
            # Optimize existing image
            temp_jpeg_path = optimize_image(file_path, is_obj=False)

        if not temp_jpeg_path:
            raise ValueError("Failed to prepare image for Vision AI")

        # 2. EXTRACT WITH AI
        p_type, items, usage = parse_items_with_llm(temp_jpeg_path)
        
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
            "temp_path": temp_jpeg_path
        }

    except Exception as e:
        logger.error(f"Error on Page {page_num}: {e}")
        return {
            "page_no": str(page_num),
            "page_type": "Bill Detail",
            "bill_items": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "temp_path": temp_jpeg_path
        }

def process_bill(file_url: str) -> dict:
    start_run = time.time()
    logger.info(f"Starting optimized bill processing for URL: {file_url}")
    local_path = download_url_to_file(file_url)
    ext = os.path.splitext(local_path)[1].lower()

    final_results = []
    total_items = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    cleanup_list = [local_path]

    try:
        tasks = []
        is_pdf = False

        if ext == '.pdf':
            is_pdf = True
            info = pdfinfo_from_path(local_path)
            max_pages = info["Pages"]
            logger.info(f"PDF has {max_pages} pages. Launching 5 parallel workers...")
            for i in range(1, max_pages + 1):
                tasks.append(i)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            logger.info("Processing single image...")
            tasks.append(1)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # --- OPTIMIZATION: 5 WORKERS ---
        # Smaller images = Less RAM = More Workers safe
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {
                executor.submit(process_page_task, p_num, local_path, is_pdf): p_num 
                for p_num in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                res = future.result()
                final_results.append(res)
                
                total_items += len(res["bill_items"])
                u = res["token_usage"]
                for k in total_usage:
                    total_usage[k] += u.get(k, 0)
                
                # Cleanup fast
                if res["temp_path"] and os.path.exists(res["temp_path"]):
                    try: os.remove(res["temp_path"])
                    except: pass

        final_results.sort(key=lambda x: int(x["page_no"]))
        
        cleaned_results = []
        for r in final_results:
            cleaned_results.append({
                "page_no": r["page_no"],
                "page_type": r["page_type"],
                "bill_items": r["bill_items"]
            })

    finally:
        for f in cleanup_list:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        gc.collect()

    total_time = time.time() - start_run
    logger.info(f"RUN COMPLETE. Time: {total_time:.2f}s. Items: {total_items}")

    return {
        "is_success": True,
        "token_usage": total_usage,
        "data": {
            "pagewise_line_items": cleaned_results,
            "total_item_count": total_items
        }
    }
