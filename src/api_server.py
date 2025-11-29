from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, validator
from loguru import logger
from src.pipeline import process_bill
import json

router = APIRouter()

class BillExtractionRequest(BaseModel):
    url: str

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        if "AIza" in v: 
            raise ValueError('Invalid URL. It looks like you submitted an API Key.')
        return v

@router.post("/extract_bill", status_code=status.HTTP_200_OK)
async def extract_bill(payload: BillExtractionRequest):
    try:
        logger.info(f"Received document URL: {payload.url}")
        
        result = process_bill(payload.url)
        
        # Log result for debugging in Render Dashboard
        logger.info(f"Extraction Complete. Found {result['data']['total_item_count']} items.")
        
        return {"status": "success", "data": result}

    except ValueError as ve:
        logger.error(f"Input Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
