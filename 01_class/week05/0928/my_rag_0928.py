# =====================================================================
# 가상Tech 업무 가이드 RAG — 총괄이 설계한 버전 (2026-09-28)
# ---------------------------------------------------------------------
# 수업 노트북(rag_02_0928)과 같은 PDF·질문 파일로, 다섯 가지를 다르게 했다.
#   ① 읽기  : 페이지 머리글을 지우고, 표는 "한 행 = 한 문장"으로 바꾼다
#   ② 조각  : 500자마다 자르지 않고 문서의 "절"(7.1 휴가 등) 단위로 자른다
#   ③ 검색  : 키워드 검색(BM25) + 의미 검색(임베딩) 두 결과를 합친다(RRF)
#   ④ 질문  : 두 가지를 묻는 질문은 LLM이 하위 질문으로 나눠 각각 검색한다
#   ⑤ 답변  : 근거 절·쪽을 적고, 규정이 없으면 "없다"고 먼저 말한다
#
# 실행 준비 (강의실 Windows)
#   pip install pymupdf rank_bm25 langchain-openai langchain-community faiss-cpu python-dotenv
#   이 파일과 같은 폴더에 .env(OPENAI_API_KEY), data/virtual_tech_internal_handbook.pdf, question.text
#   실행: python my_rag_0928.py
# =====================================================================

import re                      # 정규식: 절 제목("7.1 휴가") 같은 글자 모양을 찾는 도구
import json                    # LLM이 돌려준 하위 질문 목록(JSON 글자)을 파이썬 리스트로 바꾼다
from pathlib import Path       # 파일 경로를 OS(Windows/Mac) 상관없이 다루는 도구

import pymupdf                 # PDF를 읽는 라이브러리(수업의 PyMuPDFLoader가 안에서 쓰는 것과 같은 것)
from rank_bm25 import BM25Okapi  # 키워드 검색 알고리즘 BM25 — 질문 단어가 조각에 얼마나 자주·드물게 나오는지로 점수
from dotenv import load_dotenv, find_dotenv   # .env 파일의 키를 환경 변수로 불러온다(값을 화면에 찍지 않는다)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # LLM과 임베딩 모델
from langchain_community.vectorstores import FAISS          # 수업과 같은 벡터저장소
from langchain_core.documents import Document               # 조각 하나를 담는 LangChain 객체
from langchain_core.messages import HumanMessage            # LLM에 보낼 메시지 한 개

load_dotenv(find_dotenv(usecwd=True))   # 지금 폴더부터 위로 올라가며 .env를 찾는다(0928 폴더 위 rag_two/.env도 찾음)

PDF_PATH = Path("./data/virtual_tech_internal_handbook.pdf")   # 수업 폴더 구조 그대로
QUESTION_PATH = Path("./question.text")                          # 파일 이름 철자도 수업 그대로

# 모든 페이지 맨 위에 반복되는 머리글 목록 정의. 검색에 방해만 되므로 실제로 거르는 곳은 build_chunks() 안
HEADER_LINES = {"가상Tech 업무 가이드", "사내용 · v1.0", "문서 책임: People Operations"}


# ---------------------------------------------------------------------
# ① 읽기 — 표를 문장으로 바꾸기
# ---------------------------------------------------------------------
def table_rows_as_text(page):
    """페이지의 표를 찾아 한 행을 한 문장으로 바꾼다.

    PDF 표를 그냥 글자로 읽으면 "구분 / 신청 시점 / 승인·증빙 / 연차 / 1일 전 / ..."처럼
    한 칸씩 줄바꿈되어 "연차"와 "1일 전"이 같은 행이라는 관계가 사라진다.
    그래서 행마다 "연차: 신청 시점 1일 전 / 승인·증빙 팀장 승인, ..."으로 이어 붙인다.
    돌려주는 값: [(표가 있는 사각형 위치, 표를 바꾼 문장들), ...]
    """
    results = []
    for table in page.find_tables().tables:                 # 이 페이지에서 찾은 표 하나씩
        rows = [[(cell or "").replace("\n", " ").strip() for cell in row]
                for row in table.extract()]                  # 표를 [행][칸] 2차원 리스트로 꺼낸다
        if len(rows) < 2:                                    # 머리 행만 있는 표는 건너뛴다
            continue
        header, body = rows[0], rows[1:]                     # 첫 행은 칸 이름(구분·신청 시점·승인·증빙)
        lines = []
        for row in body:                                     # 두 번째 행부터 실제 내용
            parts = [f"{h} {v}" for h, v in zip(header[1:], row[1:]) if v]  # "신청 시점 1일 전" 식으로 칸 이름+값
            lines.append(f"{row[0]}: " + " / ".join(parts))                  # 첫 칸(연차)을 앞에 둔다
        results.append((table.bbox, "\n".join(lines)))       # bbox = 표가 차지한 사각형(위치)
    return results


def page_blocks(page):
    """표 밖의 글 덩어리와 표 문장을 위에서 아래 순서로 섞어 돌려준다."""
    tables = table_rows_as_text(page)
    items = []
    for block in page.get_text("blocks"):                    # 글 덩어리(단락) 단위로 읽기
        x0, y0, x1, y1, text = block[:5]                     # 덩어리의 위치와 글자
        inside_table = any(y0 >= tb[1] - 1 and y1 <= tb[3] + 1 for tb, _ in tables)
        if inside_table:                                     # 표 안 글자는 위에서 만든 표 문장으로 대신한다
            continue
        items.append((y0, text.strip()))                     # y0 = 위에서부터의 높이(정렬용)
    for bbox, text in tables:
        items.append((bbox[1], "[표]\n" + text))             # 표 문장도 표가 있던 높이에 끼워 넣는다
    return [text for _, text in sorted(items)]               # 높이 순으로 정렬해 글만 돌려준다


# ---------------------------------------------------------------------
# ② 조각 — 문서의 절 단위로 자르기
# ---------------------------------------------------------------------
SECTION = re.compile(r"^(\d\.\d)\s+\S")     # "7.1 휴가"처럼 숫자.숫자 + 제목으로 시작하는 줄
CHAPTER = re.compile(r"^CHAPTER\s+(\d+)")   # "CHAPTER 07"로 시작하는 줄


def build_chunks():
    """PDF 전체를 절 단위 조각 리스트로 만든다.

    조각을 나누는 기준(위에서 아래로 읽다가 이런 줄을 만나면 새 조각 시작):
      - "CHAPTER 07" 같은 장 제목
      - "7.1 휴가" 같은 절 제목
      - "Q. ..."로 시작하는 FAQ 한 문항
      - 12자 이하의 짧은 제목 줄(알림 상자: "휴가 중 연락", "운영 DB 접근" 등)
    조각 맨 앞에는 "[CHAPTER 07 휴가·비용·복지 > 7.1 휴가]"를 붙여,
    절 이름도 검색 대상이 되게 한다.
    """
    doc = pymupdf.open(PDF_PATH)
    chunks = []                 # 완성된 조각들
    chapter, section = "", ""   # 지금 읽고 있는 장·절 이름
    buffer = []                 # 아직 조각으로 묶지 않은 줄들
    page_no = 0

    def flush():
        """buffer에 모인 줄을 조각 하나로 확정한다."""
        body = "\n".join(buffer).strip()
        if body:
            head = f"[{chapter}{' > ' + section if section else ''}]"
            chunks.append({"text": f"{head}\n{body}",          # 검색·답변에 쓰는 본문
                           "page": page_no + 1,                 # 사람이 보는 쪽 번호(1부터)
                           "section": section or chapter})      # 근거 표시용 절 이름
        buffer.clear()

    for page_no, page in enumerate(doc):
        for block in page_blocks(page):
            lines = [l for l in block.splitlines()
                     if l.strip()                               # 빈 줄 제외
                     and l.strip() not in HEADER_LINES          # 머리글 제외
                     and not re.fullmatch(r"\d{2}", l.strip())] # "08" 같은 쪽 번호 제외
            if not lines:
                continue
            first = lines[0].strip()
            if CHAPTER.match(first):                            # 장이 바뀜
                flush()
                chapter = " ".join(lines[:2]).strip()           # "CHAPTER 07" + "휴가·비용·복지"
                section = ""
                lines = lines[2:]
            elif SECTION.match(first):                          # 절이 바뀜
                flush()
                section = first
                lines = lines[1:]
            elif first.startswith("Q. "):                       # FAQ 한 문항
                flush()
                section = first
            elif (len(first) <= 12 and not first.startswith(("•", "[표]"))
                  and not first.endswith(("다.", "."))):        # 짧은 제목 줄 = 알림 상자
                flush()
                section = first
                lines = lines[1:]
            buffer.extend(lines)
        flush()                                                 # 페이지 끝에서 남은 줄 정리
    return chunks


# ---------------------------------------------------------------------
# ③ 검색 — 키워드(BM25) + 의미(임베딩), 순위 합치기
# ---------------------------------------------------------------------
# 질문 말투. 처음엔 이걸 안 걸러서, "~하나요?"가 들어 있는 FAQ 조각이
# 거의 모든 질문의 1순위로 올라왔다(직접 돌려서 본 함정).
STOP = {"하나", "나요", "어떻", "떻게", "해야", "야하", "되나", "무엇", "엇인", "인가", "가요", "있나",
        "언제", "얼마", "경우", "하려", "려면", "합니", "니다", "습니", "는지", "할수", "수있", "있는",
        "하는", "에서", "으로", "까지", "어떻게", "무엇인가요", "있나요", "되나요", "하나요",
        "방식", "조치", "절차", "기준", "어떤", "각각", "알려", "려주", "주세", "세요"}


def tokens(text):
    """BM25용 단어 쪼개기. 한국어 형태소 분석기 없이 두 가지를 쓴다.
      - 공백 기준 단어: "연차를", "ExpenseHub"
      - 한글 두 글자 조각(bigram): "연차를" -> "연차", "차를"  (조사가 붙어도 "연차"가 걸리게)
    그다음 질문 말투(STOP)를 뺀다.
    """
    words = re.findall(r"[0-9A-Za-z#@.\-]+|[가-힣]+", text.lower())
    out = []
    for w in words:
        out.append(w)
        if re.fullmatch(r"[가-힣]+", w) and len(w) > 2:
            out += [w[i:i + 2] for i in range(len(w) - 1)]
    return [t for t in out if t not in STOP]


class HybridRetriever:
    """BM25와 FAISS 두 검색 결과를 RRF로 합치는 검색기."""

    def __init__(self, chunks):
        self.chunks = chunks
        # BM25: 조각마다 단어 목록을 만들어 넣는다
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
        # FAISS: 수업과 같은 벡터저장소. metadata에 조각 번호를 넣어 두 결과를 맞춰 볼 수 있게 한다
        docs = [Document(page_content=c["text"], metadata={"i": i}) for i, c in enumerate(chunks)]
        self.faiss = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-small"))

    def search(self, query, k=4):
        """RRF(Reciprocal Rank Fusion): 순위가 높을수록 1/(60+순위)만큼 점수를 준다.
        두 검색에서 모두 위에 있으면 점수가 더해져 최종 1순위가 된다.
        점수 단위가 다른 두 검색(BM25 점수 vs 거리)을 그대로 더할 수 없어서 "순위"로 합친다.
        """
        score = {}
        bm25_scores = self.bm25.get_scores(tokens(query))
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:10]
        for rank, i in enumerate(bm25_top):
            score[i] = score.get(i, 0) + 1 / (60 + rank)
        for rank, doc in enumerate(self.faiss.similarity_search(query, k=10)):
            i = doc.metadata["i"]
            score[i] = score.get(i, 0) + 1 / (60 + rank)
        best = sorted(score, key=lambda i: -score[i])[:k]
        return [self.chunks[i] for i in best]


# ---------------------------------------------------------------------
# ④ 질문 쪼개기 · ⑤ 답변
# ---------------------------------------------------------------------
SPLIT_PROMPT = """다음 질문이 서로 다른 정보를 두 개 이상 묻는다면 하위 질문으로 나눠 JSON 배열로만 답하세요.
하나만 묻는다면 원래 질문 하나만 담은 배열로 답하세요.
질문: {q}"""

ANSWER_PROMPT = """너는 가상Tech 사내 규정 안내 도우미다. 아래 [근거]만 사용해 한국어로 답한다.

질문이 묻는 항목마다 먼저 둘 중 하나로 판단한다. 여러 항목을 묻는 질문은 항목마다 따로 판단한다.
- 직접 규정: 질문이 묻는 그 항목의 기준이 [근거]에 적혀 있다.
  질문 표현이 문서와 달라도 같은 항목이면 직접 규정이다(예: 서울·제주 출장은 문서의 "국내 출장", 교육비는 "성장 지원").
- 규정 없음: 질문이 묻는 항목 자체가 [근거]에 없다.
  비슷해 보여도 적용 대상이 다른 규정(예: 다른 대상에 대한 기한)은 그 항목의 답이 아니다.

답 형식
- 직접 규정이면: 바로 답하고 끝에 (근거: 절 이름, 쪽)을 적는다. 조건(2박 등)에 대입해 계산했으면 계산했다고 밝힌다.
- 규정 없음이면: "문서에 직접 규정이 없습니다."로 시작하고, 가까운 규정이 있으면 "관련 규정:"으로 따로 적되
  그것이 질문의 답이 아니라고 밝힌다.
- 근거에 없는 숫자·절차를 만들지 않는다.

[근거]
{context}

[질문]
{q}"""


def split_question(llm, question):
    """④ 복합 질문을 하위 질문 리스트로 나눈다. 실패하면 원래 질문 하나만 쓴다."""
    raw = llm.invoke([HumanMessage(SPLIT_PROMPT.format(q=question))]).content
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()   # 모델이 ```json ... ```으로 감싸는 경우가 있어 벗긴다(직접 본 함정)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [question]


def answer(retriever, llm, question):
    """하위 질문마다 검색해 근거를 모은 뒤, 원래 질문으로 한 번에 답하게 한다."""
    sub_questions = split_question(llm, question)
    seen, context = set(), []
    for sq in sub_questions:
        for c in retriever.search(sq, k=4):
            if c["text"] not in seen:                   # 같은 조각이 두 번 들어가지 않게
                seen.add(c["text"])
                context.append(f"({c['section']}, {c['page']}쪽)\n{c['text']}")
    prompt = ANSWER_PROMPT.format(context="\n\n---\n\n".join(context), q=question)
    return sub_questions, llm.invoke([HumanMessage(prompt)]).content


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
if __name__ == "__main__":
    chunks = build_chunks()
    print(f"조각 {len(chunks)}개")

    retriever = HybridRetriever(chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    lines = QUESTION_PATH.read_text(encoding="utf-8").splitlines()
    questions = [l.split(". ", 1)[1] for l in lines if l[:1].isdigit()]   # "1. 연차를…" -> "연차를…"

    for n, q in enumerate(questions, 1):
        subs, a = answer(retriever, llm, q)
        print(f"Q{n}. {q}")
        if len(subs) > 1:
            print(f"   (하위 질문 {len(subs)}개로 나눠 검색: {subs})")
        print(f"A. {a}")
        print("-" * 60)
