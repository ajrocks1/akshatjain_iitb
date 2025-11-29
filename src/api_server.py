from fastapi import APIRouter, HTTPException, status, Request
from loguru import logger
from src.pipeline import process_bill
import json
from datetime import datetime

router = APIRouter()

# --- IN-MEMORY HISTORY STORAGE ---
# Stores the last 10 requests/responses
API_HISTORY = []

@router.get("/", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "online", "service": "Bill Extractor Vision AI"}

# --- NEW DEBUG ENDPOINT ---
@router.get("/debug/history", status_code=status.HTTP_200_OK)
def view_history():
    """
    Call this from your browser to see the last 10 API calls cleanly.
    """
    return {
        "count": len(API_HISTORY),
        "recent_logs": API_HISTORY
    }

# --- SHARED LOGIC ---
async def handle_extraction(request: Request):
    try:
        # 1. Capture Input
        try:
            body = await request.json()
            input_payload = body
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON body")

        # 2. Extract URL logic (Same as before)
        doc_url = body.get("document") or body.get("url")
        if not doc_url:
            for val in body.values():
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    doc_url = val
                    break
        
        if not doc_url:
            raise ValueError(f"No valid URL found. Keys: {list(body.keys())}")
        
        if "AIza" in doc_url:
            raise ValueError("Invalid URL: API Key detected.")

        logger.info(f"Processing: {doc_url}")
        
        # 3. Run Pipeline
        result = process_bill(doc_url)
        
        # 4. SAVE TO HISTORY (The Magic Part)
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "judge_input": input_payload,
            "your_response": result
        }
        
        # Keep only last 10 entries to save memory
        API_HISTORY.insert(0, log_entry)
        if len(API_HISTORY) > 10:
            API_HISTORY.pop()
            
        logger.info(f"Success. History updated. View at /debug/history")
        
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        # Also log failures to history so you can debug them
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "judge_input": body if 'body' in locals() else "Parse Error",
            "error": str(e)
        }
        API_HISTORY.insert(0, error_entry)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-bill-data", status_code=status.HTTP_200_OK)
async def extract_bill_data(request: Request):
    return await handle_extraction(request)

@router.post("/extract_bill", status_code=status.HTTP_200_OK)
async def extract_bill_old(request: Request):
    return await handle_extraction(request)
