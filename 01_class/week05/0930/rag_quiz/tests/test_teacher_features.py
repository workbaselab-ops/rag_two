"""교사용 기능의 비용 없는 회귀 테스트.

실제 OpenAI API를 호출하지 않고 인증, PDF 로딩, 합성 문항 생성 흐름,
23명 균등 분배, ZIP 저장, 평가기 입력 호환성을 확인한다.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from ragas.metrics.base import Metric

from streamlit.testing.v1 import AppTest

from teacher import dataset_service
from teacher.auth import verify_teacher_password
from teacher.dataset_service import DraftQuestion
from teacher import evaluator
from teacher.evaluator import load_questions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEACHER_PAGE = PROJECT_ROOT / "pages" / "2_교사용_데이터셋_생성.py"


class _FakeChatOpenAI:
    """구조화 출력 설정만 흉내 내는 API 비호출용 객체."""

    def with_structured_output(self, _schema: object) -> "_FakeChatOpenAI":
        return self


class TeacherFeatureTests(unittest.TestCase):
    def test_ragas_metrics_match_evaluate_api(self) -> None:
        """RAGAS 0.4.3 evaluate()가 허용하는 Metric 계열인지 확인한다."""

        class FakeResult:
            def to_pandas(self) -> pd.DataFrame:
                return pd.DataFrame()

        def fake_evaluate(**kwargs: object) -> FakeResult:
            metrics = kwargs["metrics"]
            self.assertTrue(all(isinstance(metric, Metric) for metric in metrics))
            self.assertEqual(
                [metric.name for metric in metrics],
                ["context_precision", "faithfulness", "answer_relevancy"],
            )
            return FakeResult()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(evaluator, "evaluate", side_effect=fake_evaluate),
        ):
            evaluator.run_ragas(object())

    def test_password_accepts_only_env_value(self) -> None:
        # 강사 원본은 실제 비밀번호를 여기 적었다. 환경 변수 방식으로 바꾸며 가짜 값으로 검사(2026-10-03).
        with patch.dict(os.environ, {"TEACHER_PASSWORD": "test-pass"}):
            self.assertTrue(verify_teacher_password("test-pass"))
            self.assertFalse(verify_teacher_password("wrong-pass"))
            self.assertFalse(verify_teacher_password(""))
        with patch.dict(os.environ, {"TEACHER_PASSWORD": ""}):
            self.assertFalse(verify_teacher_password(""))

    def test_pdf_is_loaded_into_source_units(self) -> None:
        units = dataset_service.load_source_units()
        self.assertEqual(len(units), 30)
        self.assertTrue(all(item["context"] for item in units))
        self.assertTrue(all(item["pdf_pages"] for item in units))

    @patch.dict(os.environ, {"TEACHER_PASSWORD": "test-pass"})
    def test_teacher_page_requires_password_and_shows_69_default(self) -> None:
        page = AppTest.from_file(str(TEACHER_PAGE)).run(timeout=20)
        self.assertTrue(any("교사용 페이지" in item.value for item in page.title))

        page.text_input[0].input("0000")
        page.button[0].click()
        page.run(timeout=20)
        self.assertTrue(any("올바르지 않습니다" in item.value for item in page.error))

        page.text_input[0].input("test-pass")
        page.button[0].click()
        page.run(timeout=20)
        self.assertTrue(
            any("교사용 합성 테스트 데이터셋 생성" in item.value for item in page.title)
        )
        self.assertEqual(
            [item.value for item in page.number_input],
            [23, 20260929, 1, 1, 1],
        )
        self.assertTrue(any("총 69개" in item.value for item in page.info))

    def test_mock_generation_distribution_zip_and_evaluator_compatibility(self) -> None:
        serial = 0

        def fake_generate_batch(
            _llm: object, contexts: list[dict], difficulty: str
        ) -> list[DraftQuestion]:
            nonlocal serial
            drafts: list[DraftQuestion] = []
            for context in contexts:
                serial += 1
                drafts.append(
                    DraftQuestion(
                        source_id=context["source_id"],
                        question=f"{difficulty} 모의 질문 {serial}",
                        reference_answer=f"{difficulty} 모의 정답 {serial}",
                        question_type="설명",
                    )
                )
            return drafts

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(dataset_service, "ChatOpenAI", return_value=_FakeChatOpenAI()),
            patch.object(dataset_service, "_generate_batch", side_effect=fake_generate_batch),
        ):
            pool = dataset_service.generate_question_pool(
                {"easy": 23, "medium": 23, "hard": 23}, seed=4365
            )

        self.assertEqual(len(pool), 69)
        self.assertEqual(len({item["id"] for item in pool}), 69)

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = dataset_service.distribute_and_save(
                pool,
                student_count=23,
                per_student={"easy": 1, "medium": 1, "hard": 1},
                seed=4365,
                output_root=Path(temporary_directory),
            )
            student_files = sorted((result["run_dir"] / "students").glob("*.json"))
            self.assertEqual(len(student_files), 23)

            all_ids: list[str] = []
            for student_file in student_files:
                payload = json.loads(student_file.read_text(encoding="utf-8"))
                self.assertEqual(payload["question_count"], 3)
                self.assertEqual(payload["difficulty_counts"], {"하": 1, "중": 1, "상": 1})
                self.assertTrue(
                    all("reference" not in item for item in payload["questions"])
                )
                all_ids.extend(item["id"] for item in payload["questions"])
            self.assertEqual(len(all_ids), len(set(all_ids)))

            answer_key = result["run_dir"] / "teacher" / "master_answer_key.json"
            self.assertEqual(len(load_questions(answer_key)), 69)

            with zipfile.ZipFile(result["zip_path"]) as archive:
                names = archive.namelist()
            self.assertIn("manifest.json", names)
            self.assertIn("teacher/master_answer_key.json", names)
            self.assertEqual(len([name for name in names if name.startswith("students/")]), 23)


if __name__ == "__main__":
    unittest.main()
