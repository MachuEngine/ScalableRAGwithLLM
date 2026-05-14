from fastapi import APIRouter
from pydantic import BaseModel


chat_router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    사용자 질의를 받아 RAG 파이프라인을 실행하고 응답을 반환한다.
    """
    pass
