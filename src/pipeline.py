import os
import tempfile
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from src.llm_utils import parse_items_with_llm
from loguru import logger
import gc
import asyncio
import time
import PIL.Image

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

def optimize_image(image_path_or_obj, is_obj=False):
    try:
        if is_obj:
            img = image_path_or_obj
        else:
            img = PIL.Image.open(image_path_or_obj)
            
        # Max 1024px is sweet spot for speed/accuracy
        img.thumbnail((1024, 1024))
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name, format='JPEG', quality=80) # Slightly lower quality for speed
            return tmp.name
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return None

# --- ASYNC WORKER ---
async def process_page_task(page_num: int, file_path: str, is_pdf: bool) -> dict:
    start_time = time.time()
    temp_jpeg_path = None
    
    try:
        # 1. CONVERT (Run in Thread Pool to avoid blocking async loop)
        if is_pdf:
            def convert():
                return convert_from_path(
                    file_path, 
                    first_page=page_num, 
                    last_page=page_num, 
                    fmt='jpeg', 
                    dpi=150,  # OPTIMIZATION: Lower DPI = Much Faster
                    thread_count=1
                )
            
            images = await asyncio.to_thread(convert)
            
            if not images:
                raise ValueError("No images from PDF conversion")
            
            # Optimize in thread
            temp_jpeg_path = await asyncio.to_thread(optimize_image, images[0], True)
            
            del images
            gc.collect()
        else:
            temp_jpeg_path = await asyncio.to_thread(optimize_image, file_path, False)

        if not temp_jpeg_path:
            raise ValueError("Failed to prepare image")

        # 2. EXTRACT (Async AI Call)
        # This is where the magic happens - waiting doesn't block CPU
        p_type, items, usage = await parse_items_with_llm(temp_jpeg_path)
        
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

        # Cleanup immediately
        if temp_jpeg_path and os.path.exists(temp_jpeg_path):
            try: os.remove(temp_jpeg_path)
            except: pass

        return {
            "page_no": str(page_num),
            "page_type": p_type,
            "bill_items": items,
            "token_usage": usage
        }

    except Exception as e:
        logger.error(f"Error on Page {page_num}: {e}")
        return {
            "page_no": str(page_num),
            "page_type": "Bill Detail",
            "bill_items": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

# --- MAIN ENTRY POINT (Called by api_server) ---
# Note: Since api_server calls this with `asyncio.to_thread`, 
# we need to run the async loop inside here.
def process_bill(file_url: str) -> dict:
    # Helper to run async code from sync wrapper
    return asyncio.run(process_bill_async(file_url))

async def process_bill_async(file_url: str) -> dict:
    start_run = time.time()
    logger.info(f"Starting ASYNC processing for URL: {file_url}")
    
    local_path = await asyncio.to_thread(download_url_to_file, file_url)
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
            logger.info(f"PDF has {max_pages} pages. Launching ASYNC tasks...")
            
            # Create a task for EVERY page immediately
            # Async allows massive concurrency (IO bound)
            # We limit to 5 concurrent conversions via Semaphore to protect RAM
            sem = asyncio.Semaphore(5) 
            
            async def bounded_process(p_num):
                async with sem:
                    return await process_page_task(p_num, local_path, True)

            for i in range(1, max_pages + 1):
                tasks.append(bounded_process(i))
        
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            tasks.append(process_page_task(1, local_path, False))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # RUN EVERYTHING AT ONCE
        results = await asyncio.gather(*tasks)
        final_results = list(results)

        # Aggregate Results
        for res in final_results:
            total_items += len(res["bill_items"])
            u = res["token_usage"]
            for k in total_usage:
                total_usage[k] += u.get(k, 0)

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
