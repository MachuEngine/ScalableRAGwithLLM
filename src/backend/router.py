import json
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.backend.graph import rag_graph

chat_router = APIRouter()

STEP_LABELS = {
    "retrieve": "리뷰 검색 완료",
    "generate": "답변 생성 완료",
    "check_hallucination": "환각 검증 완료",
}

INITIAL_STATE = {
    "documents": [],
    "answer": "",
    "hallucination": False,
    "iteration": 0,
}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    final_state = await rag_graph.ainvoke({**INITIAL_STATE, "query": request.query})
    return ChatResponse(
        answer=final_state["answer"],
        sources=final_state["documents"],
    )


@chat_router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        async for event in rag_graph.astream({**INITIAL_STATE, "query": request.query}):
            for node_name, node_output in event.items():
                payload: dict = {
                    "step": node_name,
                    "label": STEP_LABELS.get(node_name, node_name),
                }
                if "documents" in node_output:
                    payload["doc_count"] = len(node_output["documents"])
                if "answer" in node_output:
                    payload["answer"] = node_output["answer"]
                if "hallucination" in node_output:
                    payload["hallucination"] = node_output["hallucination"]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
