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
    try:
        data = process_document_from_url(req.document)
        pagewise = []
        for p in data.get("pagewise_line_items", []):
            page_no = str(p.get("page_no", "1"))
            page_type = p.get("page_type", "Bill Detail")
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
        resp = {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0},
            "error": str(e)
        }
    return resp
