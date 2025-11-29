import uvicorn
from fastapi import FastAPI
from src.api_server import router

app = FastAPI(title="Bill Extractor Vision AI")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
