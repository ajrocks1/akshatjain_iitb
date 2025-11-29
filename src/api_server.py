# src/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import process_bill
from loguru import logger

app = FastAPI()

class DocumentRequest(BaseModel):
    document: str

@app.post("/extract_bill")
async def extract_bill(payload: DocumentRequest):
    """
    API endpoint to receive a bill URL (PDF or image), process it, and return structured data.
    """
    file_url = payload.document
    logger.info(f"Received document URL: {file_url}")

    try:
        result = process_bill(file_url)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
