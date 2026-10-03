"""RAG 챗봇을 다른 화면에서도 사용할 수 있게 하는 FastAPI API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import ask, get_status

app = FastAPI(
    title="RAG Chatbot API",
    description="생성형 AI 윤리 가이드북 기반 질의응답 API",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2_000,
        examples=["생성형 AI의 주요 위험은?"],
    )


class Source(BaseModel):
    page: int
    content: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: list[Source]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAG Chatbot API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    status = get_status()
    return {"status": "ok" if status["pdf_ready"] else "error", **status}


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest) -> dict[str, Any]:
    try:
        return ask(data.question)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="답변 생성 중 오류가 발생했습니다.",
        ) from error
