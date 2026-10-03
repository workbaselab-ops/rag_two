"""교육생 RAG를 실행하고 채점 시스템 제출용 JSON을 만드는 기능."""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_ROOT = BASE_DIR / "submissions"

AskFunction = Callable[[str], dict[str, Any]]
ProgressCallback = Callable[[int, int, str], None]


def load_assigned_questions(data: bytes | str) -> tuple[str, list[dict[str, str]]]:
    """교사가 배포한 JSON에서 교육생 ID와 평가 문항을 읽는다.

    교육생 PC에서 RAGAS를 계산할 수 있도록 기준 답안과 문항 정보도 읽는다.
    """
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    payload = json.loads(text)
    student_id = ""
    if isinstance(payload, dict):
        student_id = str(payload.get("student_id", "")).strip()
        items = payload.get("questions")
    else:
        items = payload
    if not isinstance(items, list) or not items:
        raise ValueError("문항 파일은 비어 있지 않은 questions 배열을 포함해야 합니다.")

    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{index}번째 문항은 JSON 객체여야 합니다.")
        question_id = str(item.get("id", f"q{index:03d}")).strip()
        question = str(item.get("question", "")).strip()
        reference = str(item.get("reference", "")).strip()
        if not question_id or not question or not reference:
            raise ValueError(
                f"{index}번째 문항의 id, question 또는 reference가 비어 있습니다. "
                "교사용 페이지에서 자기평가용 문항 파일을 다시 생성해 주세요."
            )
        if question_id in seen_ids:
            raise ValueError(f"중복된 문항 ID입니다: {question_id}")
        seen_ids.add(question_id)
        questions.append(
            {
                "id": question_id,
                "question": question,
                "reference": reference,
                "difficulty": str(item.get("difficulty", "")).strip(),
                "type": str(item.get("type", "")).strip(),
            }
        )
    return student_id, questions


def _safe_student_id(student_id: str) -> str:
    """학생 ID를 파일명에 안전한 문자로 제한한다."""
    value = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", student_id.strip()).strip("_")
    if not value:
        raise ValueError("교육생 ID를 입력해 주세요.")
    return value[:80]


def build_submission(
    student_id: str,
    questions: list[dict[str, str]],
    ask_function: AskFunction,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """각 문항을 교육생의 RAG로 실행하고 제출용 결과를 구성한다."""
    safe_id = _safe_student_id(student_id)
    if not questions:
        raise ValueError("평가할 문항이 없습니다.")

    results: list[dict[str, Any]] = []
    for position, item in enumerate(questions, start=1):
        started = time.perf_counter()
        result: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
        }
        try:
            rag_result = ask_function(item["question"])
            answer = rag_result.get("answer")
            contexts = rag_result.get("contexts")
            sources = rag_result.get("sources", [])
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("RAG 답변이 비어 있습니다.")
            if not isinstance(contexts, list):
                raise ValueError("RAG 검색 문맥이 목록 형식이 아닙니다.")
            result.update(
                {
                    "answer": answer.strip(),
                    "contexts": contexts,
                    "sources": sources if isinstance(sources, list) else [],
                    "status": "success",
                    "error": None,
                }
            )
        except Exception as error:  # 문항별 실패도 제출 파일에 기록한다.
            result.update(
                {
                    "answer": "",
                    "contexts": [],
                    "sources": [],
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        if progress_callback:
            progress_callback(position, len(questions), item["id"])

    return {
        "schema_version": "1.0",
        "student_id": safe_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "question_count": len(questions),
        "success_count": sum(item["status"] == "success" for item in results),
        "error_count": sum(item["status"] == "error" for item in results),
        "results": results,
    }


def save_submission(
    submission: dict[str, Any], output_root: Path = SUBMISSION_ROOT
) -> Path:
    """제출 결과를 서버와 다운로드에 사용할 JSON 파일로 저장한다."""
    student_id = _safe_student_id(str(submission.get("student_id", "")))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_root / f"{student_id}_submission_{timestamp}.json"
    path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def save_submission_files(
    submission: dict[str, Any], output_root: Path = SUBMISSION_ROOT
) -> tuple[Path, Path]:
    """동일한 실행 결과를 제출용 JSON과 CSV로 함께 저장한다."""
    student_id = _safe_student_id(str(submission.get("student_id", "")))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{student_id}_submission_{timestamp}"
    json_path = output_root / f"{stem}.json"
    csv_path = output_root / f"{stem}.csv"

    json_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "schema_version",
        "student_id",
        "created_at",
        "question_id",
        "question",
        "answer",
        "contexts",
        "sources",
        "status",
        "error",
        "elapsed_seconds",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in submission.get("results", []):
            writer.writerow(
                {
                    "schema_version": submission.get("schema_version", "1.0"),
                    "student_id": student_id,
                    "created_at": submission.get("created_at", ""),
                    "question_id": result.get("id", ""),
                    "question": result.get("question", ""),
                    "answer": result.get("answer", ""),
                    # 목록 구조를 잃지 않도록 JSON 문자열로 CSV 셀에 저장한다.
                    "contexts": json.dumps(
                        result.get("contexts", []), ensure_ascii=False
                    ),
                    "sources": json.dumps(
                        result.get("sources", []), ensure_ascii=False
                    ),
                    "status": result.get("status", ""),
                    "error": result.get("error") or "",
                    "elapsed_seconds": result.get("elapsed_seconds", 0),
                }
            )
    return json_path, csv_path
