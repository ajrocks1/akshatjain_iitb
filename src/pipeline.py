# src/pipeline.py
import os, tempfile, requests
from pdf2image import convert_from_path
from src.ocr import ocr_page, save_ocr_json
from src.simple_parser import group_lines, parse_items, find_totals

# Note:
# DO NOT set a Windows poppler path here.
# Our Dockerfile installs poppler-utils in Linux and pdf2image will pick it automatically.
POPPLER_PATH = None   # Leave this as None

# Directories (Render & local safe)
BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
OCR_DIR = os.path.join(BASE_DIR, "data", "ocr")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

# Page counter resets per API call
def reset_counter():
    global PAGE_COUNTER
    PAGE_COUNTER = 0

def next_page_number():
    global PAGE_COUNTER
    PAGE_COUNTER += 1
    return PAGE_COUNTER


# ------------------------- DOWNLOAD FILE -------------------------

def download_url_to_file(url):
    """
    Downloads a URL into a temporary file.
    Detects extension from content-type or URL.
    Returns (local_file_path, content_type)
    """
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()

    # Determine extension
    if "pdf" in content_type:
        suffix = ".pdf"
    elif "png" in content_type:
        suffix = ".png"
    elif "jpeg" in content_type or "jpg" in content_type:
        suffix = ".jpg"
    else:
        parsed = url.split("?")[0].lower()
        if parsed.endswith(".pdf"): suffix = ".pdf"
        elif parsed.endswith(".png"): suffix = ".png"
        elif parsed.endswith((".jpg",".jpeg")): suffix = os.path.splitext(parsed)[1]
        else: suffix = ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.flush()
    tmp.close()

    return tmp.name, content_type


# ------------------------- PROCESS IMAGE -------------------------

def process_image_file(image_path, page_no=None):
    """
    Saves image with a stable name page_X.png and runs OCR + parsing.
    """
    import shutil

    if page_no is None:
        page_no = next_page_number()
    else:
        # synchronize counter
        global PAGE_COUNTER
        PAGE_COUNTER = page_no

    dest_name = f"page_{page_no}.png"
    dest_path = os.path.join(IMAGES_DIR, dest_name)

    # Copy into static images folder
    try:
        shutil.copyfile(image_path, dest_path)
    except:
        dest_path = image_path  # fallback

    # OCR
    words = ocr_page(dest_path)

    # Save OCR JSON
    save_ocr_json(words, os.path.join(OCR_DIR, f"{dest_name}.json"))

    # Parse into items
    lines = group_lines(words)
    items = parse_items(lines)
    totals = find_totals(lines)

    # Page object
    page_obj = {
        "page_no": str(page_no),
        "page_type": "Bill Detail",
        "bill_items": items
    }

    if totals:
        page_obj["detected_totals"] = totals

    return [page_obj], len(items)


# ------------------------- PROCESS PDF -------------------------

def process_pdf_file(pdf_path):
    """
    Converts PDF → images → processes each page.
    """
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    pagewise = []
    total_items = 0

    for i, page in enumerate(pages, start=1):
        temp_img_path = os.path.join(IMAGES_DIR, f"_temp_pdf_page_{i}.png")
        page.save(temp_img_path, "PNG")
        p_objs, cnt = process_image_file(temp_img_path, page_no=i)
        pagewise.extend(p_objs)
        total_items += cnt

    return pagewise, total_items


# ------------------------- MAIN ENTRY -------------------------

def process_document_from_url(url):
    """
    Master function called by API.
    """
    reset_counter()
    local_file = None

    try:
        local_file, content_type = download_url_to_file(url)

        # If image → process directly
        if content_type.startswith("image") or local_file.lower().endswith((".png",".jpg",".jpeg")):
            pagewise, total_items = process_image_file(local_file, page_no=1)

        # Else try PDF → fallback to image
        else:
            try:
                pagewise, total_items = process_pdf_file(local_file)
            except:
                pagewise, total_items = process_image_file(local_file, page_no=1)

        return {
            "pagewise_line_items": pagewise,
            "total_item_count": total_items
        }

    finally:
        if local_file and os.path.exists(local_file):
            try: os.remove(local_file)
            except: pass
