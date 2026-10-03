"""교육생용 RAG 구현 템플릿.

이 파일은 완성본이 아닙니다. ``notebook/rag_00.ipynb``에서 실습한 내용을
TODO 블록에 작성하면 ``app.py``와 교육생 평가 파일 생성 기능에서 바로
사용할 수 있습니다.

교육생은 아래 RAG 동작만 구현합니다.
1. PDF 문서 로드
2. 문서 청크 분할
3. 임베딩, Vector Store, retriever 구성
4. 검색 문맥을 사용하는 프롬프트와 LLM 구성

앱과 자동평가가 사용하는 ``PDF_PATH``, ``ask()``, ``get_status()``의 이름과
``ask()``의 구현은 제공됩니다.
"""

from __future__ import annotations

# =============================================================================
# [블록 0] 라이브러리 가져오기
# =============================================================================
# 아래 기본 라이브러리는 제공됩니다.
# rag_00.ipynb에서 사용한 LangChain Loader, Text Splitter, Embeddings,
# Vector Store, Prompt, Chat Model 등의 import 문을 이 영역에 추가하세요.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 주의:
# - `%pip`, `!pip`, `display()` 같은 노트북 전용 명령은 옮기지 않습니다.
# - 사용하지 않는 테스트·출력용 라이브러리는 추가하지 않습니다.

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# TODO 0: 노트북에서 사용한 RAG 관련 import 문을 아래에 작성하세요.


# =============================================================================
# [블록 1] 환경 변수와 문서 경로 - 제공 코드, 수정하지 않음
# =============================================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "생성형_AI윤리_가이드북_RAG시험용_30쪽.pdf"


def _require_api_key() -> None:
    """OpenAI 호출 전에 API 키 설정 여부를 확인한다."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다. 프로젝트 루트의 .env 파일에 "
            "OPENAI_API_KEY=... 형식으로 설정해 주세요."
        )


# =============================================================================
# [블록 2~5] RAG 구성 - 교육생 작성 영역
# =============================================================================
# 이 함수 안에 PDF 로드부터 retriever, prompt, llm 생성까지 작성하세요.
#
# @lru_cache를 삭제하면 질문할 때마다 PDF 전체를 다시 임베딩하여 시간과
# API 비용이 증가합니다. 데코레이터와 함수 이름은 변경하지 마세요.


@lru_cache(maxsize=1)
def _build_rag() -> tuple[Any, Any, Any]:
    """retriever, prompt, llm을 생성하여 반환한다."""
    _require_api_key()
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"시험 문서를 찾을 수 없습니다: {PDF_PATH}")

    # 교육생 공통 변수입니다. 이름을 바꾸지 말고 각 단계의 결과를 대입하세요.
    pages: Any = None
    chunks: Any = None
    embeddings: Any = None
    vector_store: Any = None
    retriever: Any = None
    prompt: Any = None
    llm: Any = None

    # -------------------------------------------------------------------------
    # TODO 1: PDF 로드
    # -------------------------------------------------------------------------
    # 목표:
    # - PDF_PATH의 문서를 LangChain Document 목록으로 불러옵니다.
    # - 결과를 pages 변수에 대입합니다.

    # PDF_PATH = "./data/생성형_AI윤리_가이드북_RAG시험용_30쪽.pdf"

    loadar = PyMuPDFLoader(PDF_PATH)
    pages = loadar.load()

    print(len(pages))                  
    print(pages[1].page_content[:200])





    # -------------------------------------------------------------------------
    # TODO 2: 문서 청크 분할
    # -------------------------------------------------------------------------
    # 목표:
    # - 페이지 문서를 적절한 chunk_size와 chunk_overlap으로 분할합니다.
    # - 결과는 Document 목록이어야 합니다.
    # - 결과를 chunks 변수에 대입합니다.
    # - 선택한 크기와 중첩값의 이유를 주석으로 설명하세요.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    # split_pages = text_splitter.split_documents(pages)

    # print(len(split_pages))

    chunks = text_splitter.split_documents(pages)





    # -------------------------------------------------------------------------
    # TODO 3: 임베딩, Vector Store, retriever 생성
    # -------------------------------------------------------------------------
    # 목표:
    # - 청크를 임베딩합니다.
    # - Vector Store를 생성합니다.
    # - 질문과 관련된 문맥을 검색할 retriever를 만듭니다.
    # - embeddings, vector_store, retriever 변수에 각각 대입합니다.
    # - 검색 문서 수(k)를 정하고 이유를 주석으로 설명하세요.

    embeddings = OpenAIEmbeddings()
    # vector_store = FAISS.from_documents(split_pages, embeddings)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k" : 4})

    # found = retriever.invoke("세계보건기구(WHO)가 의료 분야에서 생성형 AI 활용 시 주의할 점으로 지적한 내용은 무엇인가요?")
    # for i, d in enumerate(found, 1):
    #     print(i, d.metadata.get("page"), d.page_content[:60].replace("\n", " "))





    # -------------------------------------------------------------------------
    # TODO 4: 프롬프트 생성
    # -------------------------------------------------------------------------
    # 필수 조건:
    # - 검색 문맥을 받을 {context} 변수가 있어야 합니다.
    # - 사용자 질문을 받을 {question} 변수가 있어야 합니다.
    # - 문서에 없는 내용을 추측하지 않도록 지시해야 합니다.
    # - 결과를 prompt 변수에 대입합니다.

    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved [context] to answer the [question]. 
    If you don't know the [answer] in [context], just say that you don't know the [answer]. 
    Answer in Korean.

    [context]
    {context}

    [question]
    {question}

    [answer]"""
    )





    # -------------------------------------------------------------------------
    # TODO 5: Chat Model 생성 및 반환
    # -------------------------------------------------------------------------
    # 목표:
    # - 답변을 생성할 Chat Model을 만듭니다.
    # - 결과를 llm 변수에 대입합니다.

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)





    # 아래 검증과 반환 코드는 제공 코드이므로 수정하지 않습니다.
    missing = [
        name
        for name, value in (
            ("pages", pages),
            ("chunks", chunks),
            ("embeddings", embeddings),
            ("vector_store", vector_store),
            ("retriever", retriever),
            ("prompt", prompt),
            ("llm", llm),
        )
        if value is None
    ]
    if missing:
        raise NotImplementedError(
            "_build_rag()의 다음 공통 변수를 완성해 주세요: " + ", ".join(missing)
        )
    return retriever, prompt, llm


# =============================================================================
# [제공 함수] LLM 응답을 문자열로 변환 - 수정하지 않음
# =============================================================================


def _answer_text(content: Any) -> str:
    """LLM 응답의 문자열/블록형 content를 화면용 문자열로 통일한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content)


# =============================================================================
# [제공 함수] 질문 검색, 답변 생성, 표준 결과 반환 - 수정하지 않음
# =============================================================================
# app.py와 RAGAS 평가 기능이 직접 호출하는 공개 함수입니다.
# 함수 이름, question 매개변수, 반환 키를 변경하면 안 됩니다.


def ask(question: str) -> dict[str, Any]:
    """질문에 답하고 검색 문맥과 페이지 출처를 함께 반환한다."""
    question = question.strip()
    if not question:
        raise ValueError("질문을 입력해 주세요.")

    retriever, prompt, llm = _build_rag()
    documents = list(retriever.invoke(question))

    document_rows: list[tuple[Any, str]] = []
    for document in documents:
        content = str(getattr(document, "page_content", "")).strip()
        if content:
            document_rows.append((document, content))
    if not document_rows:
        raise RuntimeError("질문과 관련된 검색 문맥을 찾지 못했습니다.")

    contexts = [content for _, content in document_rows]
    context_text = "\n\n---\n\n".join(contexts)

    messages = prompt.invoke({"question": question, "context": context_text})
    response = llm.invoke(messages)
    answer = _answer_text(getattr(response, "content", response)).strip()
    if not answer:
        raise RuntimeError("LLM이 빈 답변을 반환했습니다.")

    sources: list[dict[str, Any]] = []
    for document, content in document_rows:
        metadata = getattr(document, "metadata", {}) or {}
        page_value = metadata.get("page", 0)
        try:
            page = int(page_value) + 1
        except (TypeError, ValueError):
            page = 1
        sources.append({"page": page, "content": content})

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


# =============================================================================
# [제공 함수] 화면 상태 확인 - 수정하지 않음
# =============================================================================


def get_status() -> dict[str, Any]:
    """외부 API를 호출하지 않고 PDF·API 키·색인 상태를 반환한다."""
    return {
        "pdf_ready": PDF_PATH.is_file(),
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "index_initialized": _build_rag.cache_info().currsize > 0,
    }


# =============================================================================
# [완성 후 확인]
# =============================================================================
# 1. TODO 0~5를 구현해 _build_rag()의 공통 변수에 결과를 대입합니다.
# 2. 문법을 검사합니다.
#
#      .venv\Scripts\python.exe -m py_compile rag.py
#
# 3. Streamlit을 실행해 질문, 답변, 검색 근거를 확인합니다.
#
#      uv run streamlit run app.py
#
# 4. ask() 결과의 answer는 str, contexts는 비어 있지 않은 list[str]인지
#    반드시 확인합니다. 이 규격이 맞아야 제출 파일과 RAGAS 평가가 동작합니다.
# =============================================================================


# if __name__ == "__main__":
#     _build_rag()