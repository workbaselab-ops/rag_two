"""교사용 서비스: RAGAS 자동채점기.

두 가지 입력 방식을 지원한다.

1. 기존 방식: 숨김 평가 질문으로 ``rag.ask()``를 직접 호출한다.
2. 파일 제출 방식: 교육생 제출 JSON의 답변과 검색 문맥을 교사용 정답지에
   연결한다.

두 방식 모두 RAGAS로 평가해 CSV와 JSON 요약 파일을 생성한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 이 파일을 ``python teacher/evaluator.py``로 직접 실행해도 교육생의 rag.py를
# 가져올 수 있도록 프로젝트 루트를 모듈 검색 경로에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_ragas_vertexai_compatibility() -> None:
    """RAGAS 0.4.3의 제거된 VertexAI import 문제를 우회한다.

    OpenAI 평가에서는 해당 클래스가 사용되지 않는다. RAGAS가 수정되면 이 함수와
    호출부를 제거할 수 있다.
    """
    module_name = "langchain_community.chat_models.vertexai"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            vertex_module = types.ModuleType(module_name)
            vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[module_name] = vertex_module

        import langchain_community.llms as community_llms

    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (), {})


_install_ragas_vertexai_compatibility()

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics._answer_relevance import AnswerRelevancy  # noqa: E402
from ragas.metrics._context_precision import (  # noqa: E402
    LLMContextPrecisionWithReference,
)
from ragas.metrics._faithfulness import Faithfulness  # noqa: E402


METRIC_WEIGHTS = {
    "context_precision": 0.30,
    "faithfulness": 0.40,
    "answer_relevancy": 0.30,
}


@dataclass(frozen=True)
class QuestionItem:
    question_id: str
    question: str
    reference: str
    difficulty: str
    question_type: str


def _non_empty_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{index + 1}번째 항목의 {field!r} 값이 비어 있습니다.")
    return value.strip()


def load_questions(path: Path) -> list[QuestionItem]:
    """공개/합성 데이터셋에서 평가 질문과 기준 답변을 읽는다."""
    if not path.is_file():
        raise FileNotFoundError(f"평가 질문 파일을 찾을 수 없습니다: {path}")
    if not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"평가 질문 파일이 비어 있습니다: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("questions")
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "질문 파일은 JSON 배열 또는 {'questions': [...]} 형식이어야 합니다."
        )

    questions: list[QuestionItem] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{index + 1}번째 항목은 JSON 객체여야 합니다.")

        question = _non_empty_text(
            item.get("question", item.get("user_input")), "question", index
        )
        reference = _non_empty_text(
            item.get("reference", item.get("reference_answer")),
            "reference",
            index,
        )
        question_id = str(item.get("id", f"q{index + 1:03d}")).strip()
        if not question_id:
            raise ValueError(f"{index + 1}번째 항목의 'id' 값이 비어 있습니다.")
        if question_id in seen_ids:
            raise ValueError(f"중복된 질문 ID입니다: {question_id}")
        seen_ids.add(question_id)

        questions.append(
            QuestionItem(
                question_id=question_id,
                question=question,
                reference=reference,
                difficulty=str(item.get("difficulty", "")).strip(),
                question_type=str(item.get("type", "")).strip(),
            )
        )
    return questions


def collect_rag_outputs(
    questions: list[QuestionItem], fail_fast: bool
) -> tuple[list[QuestionItem], list[dict[str, Any]], list[dict[str, Any]]]:
    """rag.ask()를 호출하고 성공 및 실패 결과를 분리한다."""
    from rag import ask

    successful_questions: list[QuestionItem] = []
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for position, item in enumerate(questions, start=1):
        print(f"[{position}/{len(questions)}] {item.question_id} 답변 생성 중...")
        started_at = time.perf_counter()
        try:
            result = ask(item.question)
            if not isinstance(result, dict):
                raise TypeError("ask() 반환값은 dict여야 합니다.")

            answer = result.get("answer")
            contexts = result.get("contexts")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("ask() 결과의 'answer'가 비어 있습니다.")
            if not isinstance(contexts, list) or not all(
                isinstance(context, str) and context.strip() for context in contexts
            ):
                raise ValueError("ask() 결과의 'contexts'는 문자열 목록이어야 합니다.")

            successful_questions.append(item)
            outputs.append(
                {
                    "answer": answer.strip(),
                    "contexts": contexts,
                    "latency_seconds": round(time.perf_counter() - started_at, 3),
                }
            )
        except Exception as error:
            failure = {
                "question_id": item.question_id,
                "question": item.question,
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            print(f"  실패: {failure['error']}")
            if fail_fast:
                raise RuntimeError(
                    f"{item.question_id} 평가 응답 생성에 실패했습니다."
                ) from error

    if not outputs:
        raise RuntimeError("평가 가능한 RAG 응답이 하나도 생성되지 않았습니다.")
    return successful_questions, outputs, failures


def collect_submission_outputs(
    questions: list[QuestionItem],
    submission_path: Path,
    *,
    fail_fast: bool = False,
    limit: int | None = None,
) -> tuple[str, list[QuestionItem], list[dict[str, Any]], list[dict[str, Any]]]:
    """교육생 제출 파일을 교사용 정답 문항에 연결해 평가 입력으로 변환한다."""
    if not submission_path.is_file():
        raise FileNotFoundError(f"교육생 제출 파일을 찾을 수 없습니다: {submission_path}")
    if submission_path.suffix.lower() == ".csv":
        with submission_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if not rows:
            raise ValueError(f"교육생 제출 CSV가 비어 있습니다: {submission_path}")
        student_ids = {str(row.get("student_id", "")).strip() for row in rows}
        if len(student_ids) != 1 or "" in student_ids:
            raise ValueError("CSV의 모든 행에는 동일한 student_id가 필요합니다.")
        results: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            try:
                contexts = json.loads(row.get("contexts", "[]"))
                sources = json.loads(row.get("sources", "[]"))
                elapsed_seconds = float(row.get("elapsed_seconds", 0))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"CSV {index}번째 행의 구조화 값이 잘못되었습니다.") from error
            results.append(
                {
                    "id": row.get("question_id", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "contexts": contexts,
                    "sources": sources,
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        payload: dict[str, Any] = {
            "student_id": next(iter(student_ids)),
            "results": results,
        }
    elif submission_path.suffix.lower() == ".json":
        raw = submission_path.read_text(encoding="utf-8-sig")
        if not raw.strip():
            raise ValueError(f"교육생 제출 파일이 비어 있습니다: {submission_path}")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("교육생 제출 파일의 최상위 값은 JSON 객체여야 합니다.")
    else:
        raise ValueError("교육생 제출 파일은 .csv 또는 .json 형식이어야 합니다.")

    student_id = _non_empty_text(payload.get("student_id"), "student_id", 0)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("교육생 제출 파일에 비어 있지 않은 results 배열이 필요합니다.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit은 1 이상의 정수여야 합니다.")
        results = results[:limit]

    question_by_id = {item.question_id: item for item in questions}
    successful_questions: list[QuestionItem] = []
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, result in enumerate(results, start=1):
        question_id = ""
        try:
            if not isinstance(result, dict):
                raise ValueError("결과 항목은 JSON 객체여야 합니다.")
            question_id = _non_empty_text(result.get("id"), "id", index - 1)
            if question_id in seen_ids:
                raise ValueError(f"중복된 제출 문항 ID입니다: {question_id}")
            seen_ids.add(question_id)
            if question_id not in question_by_id:
                raise ValueError(f"교사용 정답지에 없는 문항 ID입니다: {question_id}")

            item = question_by_id[question_id]
            submitted_question = _non_empty_text(
                result.get("question"), "question", index - 1
            )
            if submitted_question != item.question:
                raise ValueError(f"정답지와 질문 내용이 다릅니다: {question_id}")
            if result.get("status") != "success":
                raise ValueError(
                    str(result.get("error") or "교육생 RAG 실행에 실패한 문항입니다.")
                )

            answer = _non_empty_text(result.get("answer"), "answer", index - 1)
            contexts = result.get("contexts")
            if not isinstance(contexts, list) or not contexts or not all(
                isinstance(context, str) and context.strip() for context in contexts
            ):
                raise ValueError("contexts는 비어 있지 않은 문자열 목록이어야 합니다.")
            latency = result.get("elapsed_seconds", 0)
            if not isinstance(latency, (int, float)) or latency < 0:
                raise ValueError("elapsed_seconds는 0 이상의 숫자여야 합니다.")

            successful_questions.append(item)
            outputs.append(
                {
                    "answer": answer,
                    "contexts": [context.strip() for context in contexts],
                    "latency_seconds": round(float(latency), 3),
                }
            )
        except (TypeError, ValueError) as error:
            failure = {
                "question_id": question_id or f"row_{index}",
                "question": result.get("question", "") if isinstance(result, dict) else "",
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            if fail_fast:
                raise RuntimeError(
                    f"{failure['question_id']} 제출 결과 검증에 실패했습니다."
                ) from error

    if not outputs:
        raise RuntimeError("제출 파일에 평가 가능한 성공 응답이 없습니다.")
    return student_id, successful_questions, outputs, failures


def build_dataset(
    questions: list[QuestionItem], outputs: list[dict[str, Any]]
) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=item.question,
            retrieved_contexts=output["contexts"],
            response=output["answer"],
            reference=item.reference,
        )
        for item, output in zip(questions, outputs, strict=True)
    ]
    return EvaluationDataset(samples=samples)


def run_ragas(dataset: EvaluationDataset) -> pd.DataFrame:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해 주세요.")

    evaluator_model = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4.1-mini")
    embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    # RAGAS 0.4.3의 evaluate()는 legacy Metric 계열만 허용한다. 따라서
    # metrics.collections의 새 지표가 아니라 LangChain 래퍼 기반 지표를 사용한다.
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=evaluator_model, temperature=0, api_key=api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    )
    metrics = [
        LLMContextPrecisionWithReference(
            name="context_precision",
            llm=evaluator_llm,
        ),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=True,
        show_progress=True,
    )
    return result.to_pandas()


def build_score_table(
    student_id: str,
    questions: list[QuestionItem],
    outputs: list[dict[str, Any]],
    evaluated: pd.DataFrame,
) -> pd.DataFrame:
    if len(evaluated) != len(questions):
        raise RuntimeError("RAGAS 결과 행 수와 평가 질문 수가 일치하지 않습니다.")

    rows: list[dict[str, Any]] = []
    for index, (item, output) in enumerate(
        zip(questions, outputs, strict=True)
    ):
        metric_values = {
            name: float(evaluated.iloc[index][name]) for name in METRIC_WEIGHTS
        }
        ragas_score = sum(
            metric_values[name] * weight
            for name, weight in METRIC_WEIGHTS.items()
        ) * 100
        rows.append(
            {
                "student_id": student_id,
                "question_id": item.question_id,
                "difficulty": item.difficulty,
                "type": item.question_type,
                "question": item.question,
                "reference": item.reference,
                "answer": output["answer"],
                "contexts_count": len(output["contexts"]),
                "latency_seconds": output["latency_seconds"],
                **metric_values,
                "ragas_score": round(ragas_score, 2),
            }
        )
    return pd.DataFrame(rows)


def save_results(
    scores: pd.DataFrame,
    failures: list[dict[str, Any]],
    output_path: Path,
    student_id: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False, encoding="utf-8-sig")

    metric_averages = {
        metric: round(float(scores[metric].mean()), 4)
        for metric in [*METRIC_WEIGHTS, "ragas_score"]
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary = {
        "student_id": student_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "successful_questions": len(scores),
        "failed_questions": len(failures),
        "metric_weights": METRIC_WEIGHTS,
        "averages": metric_averages,
        "failures": failures,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 기반 강사용 자동채점기")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("questions.json"),
        help="평가 질문 JSON 파일 (기본값: questions.json)",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        help="교육생 평가 결과 JSON 파일. 지정하면 rag.ask()를 다시 실행하지 않음",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scores.csv"),
        help="점수 CSV 저장 경로 (기본값: scores.csv)",
    )
    parser.add_argument(
        "--student-id",
        help="결과에 기록할 교육생 ID. 미지정 시 제출 파일 ID 또는 reference_rag 사용",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="앞에서부터 평가할 문제 수",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="한 문제라도 응답 생성에 실패하면 즉시 중단",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="질문 파일만 검사하고 API를 호출하지 않음",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        questions = load_questions(args.questions)
        if args.limit is not None and args.submission is None:
            if args.limit < 1:
                raise ValueError("--limit은 1 이상의 정수여야 합니다.")
            questions = questions[: args.limit]

        if args.submission is not None:
            file_student_id, successful, outputs, failures = collect_submission_outputs(
                questions,
                args.submission,
                fail_fast=args.fail_fast,
                limit=args.limit,
            )
            student_id = args.student_id or file_student_id
            print(
                f"제출 파일 평가: {student_id} · 성공 {len(successful)}개 · "
                f"실패 {len(failures)}개"
            )
        else:
            student_id = args.student_id or "reference_rag"
            print(f"평가 질문: {len(questions)}개")
            if args.validate_only:
                print("질문 파일 검증 완료 (API 호출 없음)")
                return 0
            successful, outputs, failures = collect_rag_outputs(
                questions,
                fail_fast=args.fail_fast,
            )

        if args.validate_only:
            print("정답지와 제출 파일 검증 완료 (API 호출 없음)")
            return 0

        dataset = build_dataset(successful, outputs)
        evaluated = run_ragas(dataset)
        scores = build_score_table(
            student_id,
            successful,
            outputs,
            evaluated,
        )
        summary_path = save_results(
            scores,
            failures,
            args.output,
            student_id,
        )

        print("\n평가 완료")
        print(f"CSV: {args.output.resolve()}")
        print(f"요약: {summary_path.resolve()}")
        # print(f"평균 RAGAS 점수: {scores['ragas_score'].mean():.2f}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
