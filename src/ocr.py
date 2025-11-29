import pytesseract
from PIL import Image
from loguru import logger

def extract_text_from_image(image_path: str) -> str:
    """
    Perform OCR on an image file and return extracted text.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to open image for OCR: {e}")
        return ""
    try:
        # Use Tesseract to do OCR; adjust config if needed (e.g., language, PSM)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"OCR failed on image: {e}")
        return ""
