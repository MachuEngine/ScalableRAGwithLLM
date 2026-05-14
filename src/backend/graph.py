"""
LangGraph 기반 RAG 파이프라인.

흐름: Retrieve -> Generate -> Check_Hallucination
환각 감지 시 Generate로 되돌아가는 순환(Cyclic) 구조.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from src.backend.database import CHROMA_DIR

MAX_ITERATIONS = 3

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 영화 리뷰 분석 전문가입니다.
아래 실제 관객 리뷰들을 바탕으로 사용자의 질문에 답변하세요.
반드시 다음 형식으로 '주요 장점'과 '주요 단점'을 명확히 구분하여 요약하세요.

**주요 장점**
- (리뷰에 언급된 장점들)

**주요 단점**
- (리뷰에 언급된 단점들)

리뷰에 없는 내용은 절대 추가하지 마세요.

[참고 리뷰]
{context}"""),
    ("human", "{query}"),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """아래 [참고 리뷰]와 [생성된 답변]을 비교하세요.
답변이 리뷰 내용에만 근거하고 있으면 'yes',
리뷰에 없는 내용이 포함되어 있으면 'no'로만 답하세요.

[참고 리뷰]
{context}

[생성된 답변]
{answer}"""),
    ("human", "답변이 리뷰에 근거하고 있나요? (yes/no)"),
])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    query: str
    documents: list[str]
    answer: str
    hallucination: bool
    iteration: int


# ---------------------------------------------------------------------------
# 지연 초기화 (API 키가 로드된 후 실제 호출 시 생성)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@lru_cache(maxsize=1)
def _get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return store.as_retriever(search_kwargs={"k": 4})


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> dict:
    docs = _get_retriever().invoke(state["query"])
    return {"documents": [doc.page_content for doc in docs]}


def generate(state: RAGState) -> dict:
    chain = GENERATE_PROMPT | _get_llm() | StrOutputParser()
    context = "\n".join(f"- {doc}" for doc in state["documents"])
    answer = chain.invoke({"context": context, "query": state["query"]})
    return {"answer": answer, "iteration": state["iteration"] + 1}


def check_hallucination(state: RAGState) -> dict:
    chain = HALLUCINATION_PROMPT | _get_llm() | StrOutputParser()
    context = "\n".join(f"- {doc}" for doc in state["documents"])
    result = chain.invoke({"context": context, "answer": state["answer"]})
    hallucination = result.strip().lower().startswith("no")
    return {"hallucination": hallucination}


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------

def should_regenerate(state: RAGState) -> str:
    if state["hallucination"] and state["iteration"] < MAX_ITERATIONS:
        return "generate"
    return END


# ---------------------------------------------------------------------------
# Graph 조립
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        should_regenerate,
        {"generate": "generate", END: END},
    )

    return graph.compile()


rag_graph = build_graph()
