"""교사용 서비스: PDF 근거 합성 문항 생성, 균등 분배, ZIP 저장 기능.

화면과 인증은 교사용 Streamlit 페이지에서 담당하며 이 파일은 데이터 처리만
담당한다. 교육생용 챗봇의 RAG 실행 코드와 독립적으로 구성되어 있다.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "생성형_AI윤리_가이드북_RAG시험용_30쪽.pdf"
OUTPUT_ROOT = BASE_DIR / "evaluation" / "datasets"

DIFFICULTY_LABELS = {"easy": "하", "medium": "중", "hard": "상"}
DIFFICULTY_GUIDES = {
    "easy": "한 문단에서 명시적으로 확인 가능한 핵심 사실을 묻는다.",
    "medium": "개념의 이유, 관계, 절차를 설명하거나 간단한 사례에 적용하게 한다.",
    "hard": "여러 근거를 연결해 비교·판단·적용해야 답할 수 있게 한다.",
}
QUESTION_TYPES = ("사실검색", "설명", "적용", "비교추론")


class DraftQuestion(BaseModel):
    """OpenAI 구조화 출력으로 받을 합성 문항 한 개의 형식."""
    source_id: int = Field(description="사용한 제공 문맥의 source_id")
    question: str = Field(description="PDF만으로 답할 수 있는 독립적인 질문")
    reference_answer: str = Field(description="문맥에 충실한 모범 답안")
    question_type: str = Field(description="사실검색, 설명, 적용, 비교추론 중 하나")


class DraftBatch(BaseModel):
    """한 번의 API 호출에서 생성되는 합성 문항 묶음."""
    questions: list[DraftQuestion]


ProgressCallback = Callable[[int, int, str], None]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _source_page_map(reader: PdfReader) -> list[int | None]:
    """30쪽 편집본의 페이지를 원문 페이지 번호에 대응시킨다."""
    raw = str(reader.metadata.get("/SourcePages", "")) if reader.metadata else ""
    numbers = [int(value) for value in re.findall(r"\d+", raw)]
    if len(numbers) == len(reader.pages):
        return numbers
    return [None] * len(reader.pages)


def load_source_units(pdf_path: Path = PDF_PATH) -> list[dict]:
    """PDF를 읽고 LLM 입력에 적합한 근거 단위로 나눈다."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    original_pages = _source_page_map(reader)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    units: list[dict] = []
    for pdf_index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if len(text) < 80:
            continue
        for chunk_index, chunk in enumerate(splitter.split_text(text), start=1):
            units.append(
                {
                    "source_id": len(units) + 1,
                    "pdf_pages": [pdf_index],
                    "original_pages": (
                        [original_pages[pdf_index - 1]]
                        if original_pages[pdf_index - 1] is not None
                        else []
                    ),
                    "chunk_index": chunk_index,
                    "context": chunk,
                }
            )
    if not units:
        raise ValueError("PDF에서 문항 생성에 사용할 텍스트를 찾지 못했습니다.")
    return units


def _normalize_question(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.lower())


def _contexts_for_generation(
    units: list[dict], difficulty: str, count: int, rng: random.Random
) -> list[dict]:
    selected: list[dict] = []
    primary_units = (
        rng.sample(units, count)
        if count <= len(units)
        else [rng.choice(units) for _ in range(count)]
    )
    for first in primary_units:
        if difficulty != "hard" or len(units) == 1:
            selected.append(dict(first))
            continue
        candidates = [item for item in units if item["source_id"] != first["source_id"]]
        second = rng.choice(candidates)
        selected.append(
            {
                "source_id": first["source_id"],
                "pdf_pages": sorted(set(first["pdf_pages"] + second["pdf_pages"])),
                "original_pages": sorted(
                    set(first["original_pages"] + second["original_pages"])
                ),
                "chunk_index": first["chunk_index"],
                "context": first["context"] + "\n\n[추가 근거]\n" + second["context"],
            }
        )
    return selected


def _generate_batch(
    llm: object,
    contexts: list[dict],
    difficulty: str,
) -> list[DraftQuestion]:
    source_block = "\n\n".join(
        f"[source_id={item['source_id']}, PDF {item['pdf_pages']}쪽]\n{item['context']}"
        for item in contexts
    )
    prompt = f"""당신은 생성형 AI 윤리 교육의 평가 문항 출제자입니다.
아래 제공 문맥만 근거로 난이도 '{DIFFICULTY_LABELS[difficulty]}' 문항을 정확히 {len(contexts)}개 만드세요.

난이도 기준: {DIFFICULTY_GUIDES[difficulty]}
규칙:
- 문맥에 답이 없는 내용, 외부 지식, 의견만 요구하는 질문은 금지합니다.
- 질문끼리 표현과 학습 목표가 중복되지 않아야 합니다.
- 질문만 읽어도 무엇을 묻는지 알 수 있게 작성합니다.
- 모범 답안은 1~4문장으로 작성하고 문맥의 표현과 의미를 지킵니다.
- 각 문항은 사용한 문맥의 source_id를 정확히 기록합니다.
- question_type은 {', '.join(QUESTION_TYPES)} 중 하나입니다.

제공 문맥:
{source_block}
"""
    result = llm.invoke(prompt)
    return result.questions


def generate_question_pool(
    counts: dict[str, int],
    *,
    seed: int = 20260929,
    pdf_path: Path = PDF_PATH,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """[교사용] 난이도별 지정 수량만큼 서로 다른 합성 문항을 생성한다."""
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    invalid = set(counts) - set(DIFFICULTY_LABELS)
    if invalid or any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("counts에는 easy, medium, hard의 0 이상 정수만 지정하세요.")

    total = sum(counts.values())
    if total == 0:
        raise ValueError("생성할 문항 수가 1개 이상이어야 합니다.")

    units = load_source_units(pdf_path)
    rng = random.Random(seed)
    model = os.getenv("SYNTHETIC_DATASET_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model, temperature=0.3).with_structured_output(DraftBatch)
    pool: list[dict] = []
    seen: set[str] = set()

    for difficulty in ("easy", "medium", "hard"):
        target = counts.get(difficulty, 0)
        attempts = 0
        while sum(item["difficulty_code"] == difficulty for item in pool) < target:
            completed = sum(item["difficulty_code"] == difficulty for item in pool)
            remaining = target - completed
            batch_size = min(6, remaining)
            contexts = _contexts_for_generation(units, difficulty, batch_size, rng)
            drafts = _generate_batch(llm, contexts, difficulty)
            source_lookup = {item["source_id"]: item for item in contexts}
            for draft in drafts:
                key = _normalize_question(draft.question)
                source = source_lookup.get(draft.source_id)
                if not key or key in seen or source is None:
                    continue
                question_type = draft.question_type.strip()
                if question_type not in QUESTION_TYPES:
                    question_type = "설명"
                seen.add(key)
                pool.append(
                    {
                        "id": f"q{len(pool) + 1:04d}",
                        "question": draft.question.strip(),
                        "reference": draft.reference_answer.strip(),
                        "reference_context": source["context"],
                        "difficulty": DIFFICULTY_LABELS[difficulty],
                        "difficulty_code": difficulty,
                        "type": question_type,
                        "source_pages": source["pdf_pages"],
                        "original_pages": source["original_pages"],
                    }
                )
                if sum(item["difficulty_code"] == difficulty for item in pool) >= target:
                    break
            attempts += 1
            if attempts >= max(8, target * 3):
                raise RuntimeError(
                    f"난이도 '{DIFFICULTY_LABELS[difficulty]}' 문항을 충분히 생성하지 못했습니다."
                )
            if progress_callback:
                progress_callback(len(pool), total, f"난이도 {DIFFICULTY_LABELS[difficulty]} 생성 중")

    return pool


def _student_question(item: dict, include_answers: bool) -> dict:
    result = {
        "id": item["id"],
        "question": item["question"],
        "difficulty": item["difficulty"],
        "type": item["type"],
    }
    if include_answers:
        result.update(
            {
                "reference": item["reference"],
                "reference_context": item["reference_context"],
                "source_pages": item["source_pages"],
                "original_pages": item["original_pages"],
            }
        )
    return result


def distribute_and_save(
    pool: list[dict],
    *,
    student_count: int,
    per_student: dict[str, int],
    seed: int = 20260929,
    include_answers: bool = False,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    """[교사용] 문항을 중복 없이 균등 분배하고 결과 파일을 저장한다."""
    if student_count < 1:
        raise ValueError("교육생 수는 1명 이상이어야 합니다.")
    required = {key: student_count * per_student.get(key, 0) for key in DIFFICULTY_LABELS}
    rng = random.Random(seed)
    by_difficulty: dict[str, list[dict]] = {}
    for difficulty, needed in required.items():
        items = [item for item in pool if item.get("difficulty_code") == difficulty]
        if len(items) < needed:
            raise ValueError(
                f"난이도 {DIFFICULTY_LABELS[difficulty]} 문항이 {needed}개 필요하지만 {len(items)}개뿐입니다."
            )
        rng.shuffle(items)
        by_difficulty[difficulty] = items[:needed]

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"run_{timestamp}"
    teacher_dir = run_dir / "teacher"
    students_dir = run_dir / "students"
    teacher_dir.mkdir(parents=True)
    students_dir.mkdir()

    allocations: list[dict] = []
    for student_index in range(student_count):
        questions: list[dict] = []
        difficulty_counts: dict[str, int] = {}
        for difficulty in DIFFICULTY_LABELS:
            quantity = per_student.get(difficulty, 0)
            start = student_index * quantity
            chosen = by_difficulty[difficulty][start : start + quantity]
            questions.extend(_student_question(item, include_answers) for item in chosen)
            difficulty_counts[DIFFICULTY_LABELS[difficulty]] = len(chosen)
        rng.shuffle(questions)
        student_id = f"student_{student_index + 1:02d}"
        payload = {
            "student_id": student_id,
            "question_count": len(questions),
            "difficulty_counts": difficulty_counts,
            "questions": questions,
        }
        filename = f"{student_id}_questions.json"
        (students_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        allocations.append(
            {
                "student_id": student_id,
                "file": f"students/{filename}",
                "question_ids": [item["id"] for item in questions],
                "difficulty_counts": difficulty_counts,
            }
        )

    used_ids = {item_id for row in allocations for item_id in row["question_ids"]}
    answer_key = {
        "source_pdf": PDF_PATH.name,
        "question_count": len(used_ids),
        "questions": [item for item in pool if item["id"] in used_ids],
    }
    (teacher_dir / "master_answer_key.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_pdf": PDF_PATH.name,
        "student_count": student_count,
        "questions_per_student": sum(per_student.values()),
        "total_unique_questions": len(used_ids),
        "include_answers_in_student_files": include_answers,
        "distribution": allocations,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    zip_path = output_root / f"synthetic_dataset_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))

    return {"run_dir": run_dir, "zip_path": zip_path, "manifest": manifest}


def generate_and_distribute(
    *,
    student_count: int,
    per_student: dict[str, int],
    seed: int = 20260929,
    include_answers: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """[교사용] UI에서 호출하는 문항 생성·파일 분배 단일 진입점."""
    counts = {
        difficulty: student_count * per_student.get(difficulty, 0)
        for difficulty in DIFFICULTY_LABELS
    }
    pool = generate_question_pool(
        counts, seed=seed, progress_callback=progress_callback
    )
    return distribute_and_save(
        pool,
        student_count=student_count,
        per_student=per_student,
        seed=seed,
        include_answers=include_answers,
    )


def zip_bytes(path: Path) -> bytes:
    """Streamlit 다운로드 버튼용 ZIP 바이트를 반환한다."""
    with path.open("rb") as file:
        return file.read()


def read_zip_names(data: bytes) -> list[str]:
    """테스트와 진단을 위한 ZIP 파일 목록 반환 함수."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.namelist()


def remove_run(result: dict) -> None:
    """테스트에서 만든 결과물을 정리한다."""
    shutil.rmtree(result["run_dir"], ignore_errors=True)
    result["zip_path"].unlink(missing_ok=True)
