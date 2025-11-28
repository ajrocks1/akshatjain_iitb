# src/api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import process_document_from_url
from typing import Optional

app = FastAPI(title="Bill Extraction API")

class DocumentReq(BaseModel):
    document: str

@app.post("/extract-bill-data")
def extract_bill_data(req: DocumentReq):
    """
    Wrapper to call pipeline and return the response in the exact required schema:
    {
      "is_success": boolean,
      "token_usage": {"total_tokens": int, "input_tokens": int, "output_tokens": int},
      "data": { "pagewise_line_items": [...], "total_item_count": int }
    }
    """
    try:
        # Run processing (this returns {"pagewise_line_items": [...], "total_item_count": n})
        data = process_document_from_url(req.document)

        # Ensure types: page_no must be string, page_type present, bill_items have required keys
        pagewise = []
        for p in data.get("pagewise_line_items", []):
            page_no = str(p.get("page_no", "1"))
            page_type = p.get("page_type", "Bill Detail")
            # ensure each bill_item has all keys
            clean_items = []
            for it in p.get("bill_items", []):
                clean_items.append({
                    "item_name": it.get("item_name") if it.get("item_name") is not None else "",
                    "item_amount": float(it.get("item_amount")) if it.get("item_amount") is not None else None,
                    "item_rate": float(it.get("item_rate")) if it.get("item_rate") is not None else None,
                    "item_quantity": float(it.get("item_quantity")) if it.get("item_quantity") is not None else None
                })
            pagewise.append({
                "page_no": page_no,
                "page_type": page_type,
                "bill_items": clean_items
            })

        resp = {
            "is_success": True,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": pagewise,
                "total_item_count": int(data.get("total_item_count", 0))
            }
        }
    except Exception as e:
        # on failure, return canonical schema with is_success False
        resp = {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0},
            "error": str(e)
        }
    return resp
