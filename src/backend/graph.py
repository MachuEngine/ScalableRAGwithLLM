"""
LangGraph 기반 RAG 파이프라인.

흐름: Retrieve -> Generate -> Check_Hallucination
환각 감지 시 Generate로 되돌아가는 순환(Cyclic) 구조.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    query: str               # 사용자 질의
    documents: list[str]     # 검색된 문서 청크
    answer: str              # 생성된 답변
    hallucination: bool      # 환각 감지 여부
    iteration: int           # 재생성 횟수 (무한 루프 방지용)


MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """벡터 DB에서 관련 문서를 검색한다."""
    pass


def generate(state: RAGState) -> RAGState:
    """검색된 문서를 바탕으로 LLM 응답을 생성한다."""
    pass


def check_hallucination(state: RAGState) -> RAGState:
    """생성된 답변이 문서에 근거하는지 검증한다."""
    pass


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------

def should_regenerate(state: RAGState) -> str:
    """
    환각이 감지되고 재시도 횟수가 남아 있으면 generate로,
    그렇지 않으면 종료한다.
    """
    if state["hallucination"] and state["iteration"] < MAX_ITERATIONS:
        return "generate"
    return END


# ---------------------------------------------------------------------------
# Graph 조립
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
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
