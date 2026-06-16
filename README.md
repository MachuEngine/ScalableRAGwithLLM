# ScalableRAGwithLLM

LangGraph 기반 Hallucination 제어 RAG 파이프라인 구현 프로젝트

## 개요

영화 리뷰 데이터를 ChromaDB에 벡터 스토어로 구축하고, LangGraph의 Cyclic Graph 구조를 활용해 LLM 응답의 Hallucination을 자동 감지 및 재생성하는 RAG 서비스입니다.

## 주요 기능

- **RAG 파이프라인**: Retrieve → Generate → Check Hallucination 순환 구조
- **Hallucination 제어**: LLM as a Judge 방식으로 응답 검증, 감지 시 최대 3회 재생성
- **스트리밍 응답**: FastAPI SSE 기반 단계별 처리 상태 실시간 전달
- **대화형 UI**: Chainlit 기반 채팅 인터페이스

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | FastAPI, LangGraph, LangChain |
| VectorDB | ChromaDB |
| Embedding | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| Frontend | Chainlit |
| Infra | Docker Compose, DVC |

## 아키텍처

```
사용자 질문
    ↓
[Retrieve] ChromaDB에서 관련 문서 검색
    ↓
[Generate] LLM으로 답변 생성
    ↓
[Check Hallucination] LLM as a Judge로 검증
    ↓ (Hallucination 감지 시 Generate로 재시도, 최대 3회)
최종 답변 반환
```

## 실행 방법

```bash
# 환경 변수 설정
cp .env.example .env
# OPENAI_API_KEY 입력

# 데이터 수집 및 벡터 스토어 구축
python scripts/download_data.py
python -m src.backend.database

# Docker Compose로 서비스 실행
cd src
docker-compose up --build
```
