import json

import chainlit as cl
import httpx

# BACKEND_URL = "http://localhost:8000/api/chat/stream"
BACKEND_URL = "http://backend:8000/api/chat/stream"

STEP_ICONS = {
    "retrieve": "검색",
    "generate": "생성",
    "check_hallucination": "검증",
}


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="안녕하세요! 영화 리뷰 RAG 서비스입니다.\n궁금한 영화에 대해 질문해 보세요."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    answer = ""

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            BACKEND_URL,
            json={"query": message.content},
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])
                step_name = data.get("step", "")
                label = data.get("label", step_name)
                icon = STEP_ICONS.get(step_name, "처리")

                async with cl.Step(name=f"[{icon}] {label}") as step:
                    if step_name == "retrieve":
                        step.output = f"{data.get('doc_count', 0)}개 관련 리뷰를 찾았습니다."
                    elif step_name == "generate":
                        answer = data.get("answer", "")
                        step.output = answer
                    elif step_name == "check_hallucination":
                        is_hallucination = data.get("hallucination", False)
                        step.output = "환각 감지 — 답변을 재생성합니다." if is_hallucination else "검증 통과"

    await cl.Message(content=answer).send()
