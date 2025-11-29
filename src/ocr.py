# src/ocr.py
from PIL import Image
import pytesseract
from loguru import logger

def extract_text_from_image(image_path: str) -> str:
    try:
        logger.info(f"Running OCR on {image_path}")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        logger.debug(f"OCR extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"OCR failed on {image_path}: {e}")
        return ""
