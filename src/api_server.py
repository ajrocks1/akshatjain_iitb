from fastapi import APIRouter, HTTPException, status, Request, Body
from loguru import logger
from src.pipeline import process_bill
import json
import asyncio  # <--- NEW IMPORT
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

@router.delete("/debug/history", status_code=status.HTTP_200_OK)
def clear_history():
    API_HISTORY.clear()
    return {"status": "success", "message": "History has been cleared."}

# --- SHARED LOGIC ---
async def process_extraction_logic(body: Dict[str, Any]): # Made async
    try:
        logger.info(f"INCOMING PAYLOAD: {json.dumps(body)}")
        
        doc_url = body.get("document") or body.get("url") or body.get("link") or body.get("file")
        
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
        
        # --- NON-BLOCKING CALL (The Optimization) ---
        # This moves the heavy processing to a separate thread,
        # keeping your /debug/history endpoint instant!
        result = await asyncio.to_thread(process_bill, doc_url)
        
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


@router.post("/extract-bill-data", status_code=status.HTTP_200_OK)
async def extract_bill_data(
    payload: Dict[str, Any] = Body(
        ..., 
        example={"document": "https://hackrx.blob.core.windows.net/sample.png"}
    )
):
    return await process_extraction_logic(payload)


@router.post("/extract_bill", status_code=status.HTTP_200_OK)
async def extract_bill_old(
    payload: Dict[str, Any] = Body(
        ..., 
        example={"url": "https://hackrx.blob.core.windows.net/sample.png"}
    )
):
    return await process_extraction_logic(payload)
