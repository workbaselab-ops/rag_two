from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import build_rag_chain    # ① rag_core.py의 함수를 가져온다

app = FastAPI()                         # ② 서버 객체. uvicorn server:app 의 app이 이것
rag_chain = build_rag_chain()           # ③ 서버가 켜질 때 한 번만 체인을 만든다(PDF 읽기·임베딩)


class Question(BaseModel):              # ④ 요청 본문 모양: {"question": "..."}
    question: str


@app.post("/ask")                       # ⑤ POST /ask 로 요청이 오면 아래 함수 실행
def ask(q: Question):
    answer = rag_chain.invoke(q.question)   # ⑥ 연결: 받은 질문을 RAG 체인에 넘긴다
    return {"question": q.question, "answer": answer}   # ⑦ JSON으로 돌려준다
