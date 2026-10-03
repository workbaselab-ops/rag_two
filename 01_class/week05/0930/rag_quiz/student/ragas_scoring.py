"""교육생 PC에서 제출 응답의 RAGAS 점수를 계산하고 저장한다."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from student.submission_service import SUBMISSION_ROOT
from teacher.evaluator import (
    QuestionItem,
    build_dataset,
    build_score_table,
    run_ragas,
    save_results,
)


def evaluate_submission(
    questions: list[dict[str, str]], submission: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """성공한 RAG 응답의 RAGAS 점수를 계산한다."""
    question_by_id = {item["id"]: item for item in questions}
    evaluation_questions: list[QuestionItem] = []
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for result in submission.get("results", []):
        question_id = str(result.get("id", ""))
        source = question_by_id.get(question_id)
        if source is None:
            failures.append(
                {
                    "question_id": question_id,
                    "question": result.get("question", ""),
                    "error": "배정 문항에서 찾을 수 없는 ID",
                }
            )
            continue
        if result.get("status") != "success":
            failures.append(
                {
                    "question_id": question_id,
                    "question": source["question"],
                    "error": result.get("error") or "RAG 실행 실패",
                }
            )
            continue
        evaluation_questions.append(
            QuestionItem(
                question_id=question_id,
                question=source["question"],
                reference=source["reference"],
                difficulty=source.get("difficulty", ""),
                question_type=source.get("type", ""),
            )
        )
        outputs.append(
            {
                "answer": result["answer"],
                "contexts": result["contexts"],
                "latency_seconds": result.get("elapsed_seconds", 0),
            }
        )

    if not outputs:
        raise RuntimeError("RAGAS로 평가할 성공 응답이 없습니다.")
    dataset = build_dataset(evaluation_questions, outputs)
    evaluated = run_ragas(dataset)
    scores = build_score_table(
        str(submission["student_id"]), evaluation_questions, outputs, evaluated
    )
    return scores, failures


def save_score_files(
    scores: pd.DataFrame,
    failures: list[dict[str, Any]],
    student_id: str,
    output_root: Path = SUBMISSION_ROOT,
) -> tuple[Path, Path]:
    """문항별 점수 CSV와 평균 점수 JSON을 저장한다."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    score_path = output_root / f"{student_id}_ragas_scores_{timestamp}.csv"
    summary_path = save_results(scores, failures, score_path, student_id)
    return score_path, summary_path


def create_submission_package(
    student_id: str,
    files: list[Path],
    output_root: Path = SUBMISSION_ROOT,
) -> Path:
    """답변과 RAGAS 점수 파일을 하나의 제출용 ZIP으로 묶는다."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    package_path = output_root / f"{student_id}_final_submission_{timestamp}.zip"
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    return package_path


def score_summary(path: Path) -> dict[str, Any]:
    """화면 표시를 위해 RAGAS 요약 JSON을 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))
