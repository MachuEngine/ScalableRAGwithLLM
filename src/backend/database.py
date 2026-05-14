from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
TRAIN_PARQUET = str(Path(__file__).parent.parent.parent / "data" / "raw" / "train.parquet")


def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def ingest_data(sample_size: int = 1000) -> None:
    df = pd.read_parquet(TRAIN_PARQUET).dropna().head(sample_size)

    documents = [
        Document(
            page_content=row["document"],
            metadata={"label": int(row["label"]), "id": str(row["id"])},
        )
        for _, row in df.iterrows()
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"ChromaDB에 {len(documents)}개 문서 저장 완료: {CHROMA_DIR}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_data()
