from fastapi import APIRouter, HTTPException, status, Request, Body
from loguru import logger
from src.pipeline import process_bill
import json
from datetime import datetime
from typing import Dict, Any

router = APIRouter()

# --- IN-MEMORY HISTORY ---
API_HISTORY = []

@router.get("/", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "online", "service": "Bill Extractor Vision AI"}

@router.get("/debug/history", status_code=status.HTTP_200_OK)
def view_history():
    return {
        "count": len(API_HISTORY),
        "recent_logs": API_HISTORY
    }

# --- SHARED LOGIC ---
def process_extraction_logic(body: Dict[str, Any]):
    """
    Common logic that accepts a Dictionary (parsed JSON) 
    and finds the URL regardless of the key name.
    """
    try:
        # 1. Log the exact input
        logger.info(f"INCOMING PAYLOAD: {json.dumps(body)}")
        
        # 2. Smart Key Detection
        # Check for 'document', then 'url', then 'link', then 'file'
        doc_url = body.get("document") or body.get("url") or body.get("link") or body.get("file")
        
        # Fallback: Look for ANY value that starts with http
        if not doc_url:
            for val in body.values():
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    doc_url = val
                    break
        
        if not doc_url:
            raise ValueError(f"No valid URL found. Keys received: {list(body.keys())}")

        if "AIza" in doc_url:
            raise ValueError("Invalid URL: API Key detected.")

        logger.info(f"Processing: {doc_url}")
        
        # 3. Run Pipeline
        result = process_bill(doc_url)
        
        # 4. Save History
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": body,
            "response": result
        }
        API_HISTORY.insert(0, log_entry)
        if len(API_HISTORY) > 10:
            API_HISTORY.pop()
            
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": body,
            "error": str(e)
        }
        API_HISTORY.insert(0, error_entry)
        raise HTTPException(status_code=500, detail=str(e))


# --- ROUTE 1: Official Endpoint (Restored UI) ---
@router.post("/extract-bill-data", status_code=status.HTTP_200_OK)
def extract_bill_data(
    payload: Dict[str, Any] = Body(
        ..., 
        example={"document": "https://hackrx.blob.core.windows.net/sample.png"}
    )
):
    """
    Smart Endpoint: Accepts any JSON. 
    Swagger UI will show an example input box.
    """
    return process_extraction_logic(payload)


# --- ROUTE 2: Legacy Endpoint ---
@router.post("/extract_bill", status_code=status.HTTP_200_OK)
def extract_bill_old(
    payload: Dict[str, Any] = Body(
        ..., 
        example={"url": "https://hackrx.blob.core.windows.net/sample.png"}
    )
):
    return process_extraction_logic(payload)
