from fastapi import FastAPI, UploadFile, File, HTTPException
from pipeline import process_bill
from loguru import logger
import shutil
import os

app = FastAPI()

@app.post("/extract_bill")
async def extract_bill(file: UploadFile = File(...)):
    """
    API endpoint to receive a bill file (PDF or image), process it, and return structured data.
    """
    logger.info(f"Received file: {file.filename}")
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
        logger.warning(f"Unsupported file extension: {ext}")
        raise HTTPException(status_code=400, detail="File must be PDF or image")

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {temp_path}")

        # Process the file
        result = process_bill(temp_path)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the saved file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")

    return result
