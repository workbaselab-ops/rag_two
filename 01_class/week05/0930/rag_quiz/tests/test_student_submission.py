"""교육생 제출 파일 생성 기능의 API 비호출 테스트."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from student.submission_service import (
    build_submission,
    load_assigned_questions,
    save_submission,
    save_submission_files,
)
from student.ragas_scoring import (
    create_submission_package,
    evaluate_submission,
    save_score_files,
)
from teacher.evaluator import collect_submission_outputs, load_questions


class StudentSubmissionTests(unittest.TestCase):
    def test_load_run_and_save_submission(self) -> None:
        assigned = {
            "student_id": "student_01",
            "questions": [
                {"id": "q001", "question": "첫 번째 질문", "reference": "숨길 정답"},
                {"id": "q002", "question": "두 번째 질문", "reference": "숨길 정답"},
                {"id": "q003", "question": "세 번째 질문", "reference": "숨길 정답"},
            ],
        }
        student_id, questions = load_assigned_questions(
            json.dumps(assigned, ensure_ascii=False).encode("utf-8")
        )
        self.assertEqual(student_id, "student_01")
        self.assertEqual(questions[0]["reference"], "숨길 정답")

        def fake_ask(question: str) -> dict:
            return {
                "answer": f"{question}의 답변",
                "contexts": [f"{question}의 검색 문맥"],
                "sources": [{"page": 1, "content": "검색 문맥"}],
            }

        submission = build_submission(student_id, questions, fake_ask)
        self.assertEqual(submission["question_count"], 3)
        self.assertEqual(submission["success_count"], 3)
        self.assertEqual(submission["error_count"], 0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = save_submission(submission, Path(temporary_directory))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["student_id"], "student_01")
            self.assertEqual(len(saved["results"]), 3)
            self.assertTrue(path.name.startswith("student_01_submission_"))

            # 파일 제출 시스템에서 교사용 정답지와 제출 결과를 ID로 연결한다.
            answer_key = {
                "questions": [
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "reference": f"{item['question']}의 모범 답안",
                        "difficulty": "중",
                        "type": "설명",
                    }
                    for item in questions
                ]
            }
            answer_key_path = Path(temporary_directory) / "answer_key.json"
            answer_key_path.write_text(
                json.dumps(answer_key, ensure_ascii=False), encoding="utf-8"
            )
            matched_student_id, matched, outputs, failures = collect_submission_outputs(
                load_questions(answer_key_path), path
            )
            self.assertEqual(matched_student_id, "student_01")
            self.assertEqual(len(matched), 3)
            self.assertEqual(len(outputs), 3)
            self.assertEqual(failures, [])

            # 실제 교육생 버튼과 동일하게 JSON·CSV를 동시에 저장하고 CSV도
            # evaluator.py가 같은 평가 입력으로 읽는지 확인한다.
            _json_path, csv_path = save_submission_files(
                submission, Path(temporary_directory)
            )
            csv_student_id, csv_questions, csv_outputs, csv_failures = (
                collect_submission_outputs(load_questions(answer_key_path), csv_path)
            )
            self.assertEqual(csv_student_id, "student_01")
            self.assertEqual(len(csv_questions), 3)
            self.assertEqual(len(csv_outputs), 3)
            self.assertEqual(csv_failures, [])

            fake_metrics = pd.DataFrame(
                {
                    "context_precision": [0.8, 0.7, 0.9],
                    "faithfulness": [0.9, 0.8, 1.0],
                    "answer_relevancy": [0.7, 0.9, 0.8],
                }
            )
            with patch("student.ragas_scoring.run_ragas", return_value=fake_metrics):
                scores, score_failures = evaluate_submission(questions, submission)
            self.assertEqual(len(scores), 3)
            self.assertIn("ragas_score", scores.columns)

            score_path, summary_path = save_score_files(
                scores,
                score_failures,
                "student_01",
                Path(temporary_directory),
            )
            package_path = create_submission_package(
                "student_01",
                [csv_path, _json_path, score_path, summary_path],
                Path(temporary_directory),
            )
            with zipfile.ZipFile(package_path) as archive:
                package_names = archive.namelist()
            self.assertEqual(len(package_names), 4)
            self.assertIn(score_path.name, package_names)

    def test_question_failure_is_recorded_without_stopping(self) -> None:
        questions = [
            {"id": "q001", "question": "성공 질문"},
            {"id": "q002", "question": "실패 질문"},
        ]

        def partly_failing_ask(question: str) -> dict:
            if question == "실패 질문":
                raise RuntimeError("테스트 오류")
            return {"answer": "정상 답변", "contexts": ["정상 문맥"], "sources": []}

        submission = build_submission("student_02", questions, partly_failing_ask)
        self.assertEqual(submission["success_count"], 1)
        self.assertEqual(submission["error_count"], 1)
        self.assertEqual(submission["results"][1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
