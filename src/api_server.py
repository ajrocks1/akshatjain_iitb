from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import process_bill
from loguru import logger
import requests
import shutil
import os
import uuid

app = FastAPI()

class DocumentRequest(BaseModel):
    document: str  # This will be the URL to the file

@app.post("/extract_bill")
async def extract_bill(request: DocumentRequest):
    """
    API endpoint to receive a document URL (PDF or image), download it, process it, and return structured data.
    """
    file_url = request.document
    logger.info(f"Received document URL: {file_url}")

    try:
        # Download the file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Determine file extension from content-type or URL
        content_type = response.headers.get("content-type", "").lower()
        ext = ""
        if "pdf" in content_type:
            ext = ".pdf"
        elif "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        else:
            ext = os.path.splitext(file_url)[-1].lower()
            if ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
                logger.warning(f"Unsupported file type: {content_type}")
                raise HTTPException(status_code=400, detail="Unsupported file type")

        temp_filename = f"/tmp/{uuid.uuid4()}{ext}"
        with open(temp_filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        logger.info(f"Downloaded file to: {temp_filename}")

        # Process the bill
        result = process_bill(temp_filename)
        logger.info("Bill processing successful")

    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            logger.info(f"Removed temporary file: {temp_filename}")

    return result
