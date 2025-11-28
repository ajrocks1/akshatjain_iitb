# src/ocr.py
import pytesseract, cv2, json, os

# Ensure this path matches your tesseract.exe location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_page(img_path):
    """
    Run tesseract word-level OCR and return list of word dictionaries.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    n = len(data.get('text', []))
    for i in range(n):
        txt = (data['text'][i] or "").strip()
        if not txt:
            continue
        # try convert conf to float
        conf = None
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = None
        entry = {
            "text": txt,
            "conf": conf,
            "left": int(data.get('left', [0])[i]),
            "top": int(data.get('top', [0])[i]),
            "width": int(data.get('width', [0])[i]),
            "height": int(data.get('height', [0])[i]),
            "right": int(data.get('left', [0])[i] + data.get('width', [0])[i]),
            "bottom": int(data.get('top', [0])[i] + data.get('height', [0])[i])
        }
        words.append(entry)
    return words

def save_ocr_json(words, out_json):
    """
    Save OCR word list to JSON (used for debugging).
    """
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
