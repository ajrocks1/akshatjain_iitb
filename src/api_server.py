from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, validator
from loguru import logger
from src.pipeline import process_bill

router = APIRouter()

class BillRequest(BaseModel):
    document: str  # CHANGED: 'url' -> 'document'

    @validator('document')
    def validate_document(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must start with http:// or https://')
        return v

@router.post("/extract-bill-data", status_code=status.HTTP_200_OK)  # CHANGED: Endpoint Path
async def extract_bill_data(payload: BillRequest):
    try:
        logger.info(f"Received document URL: {payload.document}")
        
        # Pass the correct field to the pipeline
        result = process_bill(payload.document)
        
        logger.info(f"Extraction Complete. Found {result['data']['total_item_count']} items.")
        
        return result

    except ValueError as ve:
        logger.error(f"Input Validation Error: {ve}")
        # Return 400 or 422 depending on preference, but 422 is standard for validation
        raise HTTPException(status_code=422, detail=str(ve))
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
