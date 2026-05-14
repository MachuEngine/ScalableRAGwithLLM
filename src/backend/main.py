from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.router import chat_router


app = FastAPI(title="영화 리뷰 RAG 서비스", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
