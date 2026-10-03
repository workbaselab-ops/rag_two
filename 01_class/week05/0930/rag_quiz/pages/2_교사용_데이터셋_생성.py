"""교사용 UI: 인증 후 합성 평가 문항 생성·분배 기능을 제공한다.

교육생용 챗봇 화면은 ``app.py``에서 담당하며, 이 파일은 교사용 페이지다.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from teacher.auth import require_teacher_login

st.set_page_config(page_title="교사용 합성 데이터셋", page_icon="🧪", layout="wide")

# 인증 전에는 생성기 모듈과 교사용 화면을 노출하지 않는다.
if not require_teacher_login():
    st.stop()

# 교사용 인증을 통과한 뒤에만 데이터셋 생성 기능을 불러온다.
from teacher.dataset_service import (  # noqa: E402
    PDF_PATH,
    generate_and_distribute,
    zip_bytes,
)

load_dotenv()

st.title("🧪 교사용 합성 테스트 데이터셋 생성")
st.caption("30쪽 가이드북을 근거로 문항을 만들고 교육생별 JSON 파일로 균등 분배합니다.")

with st.expander("저장 파일 구성", expanded=False):
    st.code(
        """synthetic_dataset_YYYYMMDD_HHMMSS.zip
├── manifest.json
├── teacher/master_answer_key.json
└── students/
    ├── student_01_questions.json
    ├── student_02_questions.json
    └── ...""",
        language="text",
    )
    st.write("RAGAS 자기평가를 위해 교육생용 파일에도 기준 답안과 근거를 포함합니다.")

# 교사용 합성 데이터셋 설정 영역.
with st.form("dataset_options"):
    left, middle, right = st.columns(3)
    with left:
        student_count = st.number_input(
            "교육생 수", min_value=1, max_value=200, value=23, step=1
        )
        seed = st.number_input(
            "재현용 시드", min_value=0, value=20260929, step=1
        )
    with middle:
        easy_count = st.number_input(
            "1인당 난이도 하", min_value=0, max_value=50, value=1, step=1
        )
        medium_count = st.number_input(
            "1인당 난이도 중", min_value=0, max_value=50, value=1, step=1
        )
    with right:
        hard_count = st.number_input(
            "1인당 난이도 상", min_value=0, max_value=50, value=1, step=1
        )
        # 교육생 PC에서 Context Precision을 계산하려면 기준 답안이 필요하다.
        include_answers = True
        st.warning("RAGAS 자기평가를 위해 교육생 파일에 기준 답안·근거가 포함됩니다.")

    per_student_total = int(easy_count + medium_count + hard_count)
    total_questions = int(student_count) * per_student_total
    st.info(
        f"교육생 파일 {int(student_count)}개 · 1인당 {per_student_total}문항 · "
        f"서로 다른 문항 총 {total_questions}개를 생성합니다."
    )
    st.caption("문항 수가 많을수록 OpenAI API 사용 시간과 비용이 증가합니다.")
    submitted = st.form_submit_button(
        "합성 데이터셋 생성 및 균등 분배",
        type="primary",
        use_container_width=True,
    )

# 생성 버튼을 누른 경우에만 OpenAI API를 호출하고 파일을 저장한다.
if submitted:
    if per_student_total == 0:
        st.error("1인당 문항 수를 1개 이상 지정해 주세요.")
    elif not PDF_PATH.is_file():
        st.error(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")
    else:
        progress = st.progress(0, text="PDF를 읽고 문항 생성을 준비합니다.")

        def update_progress(completed: int, total: int, message: str) -> None:
            ratio = min(completed / total, 1.0) if total else 1.0
            progress.progress(ratio, text=f"{message} ({completed}/{total})")

        try:
            result = generate_and_distribute(
                student_count=int(student_count),
                per_student={
                    "easy": int(easy_count),
                    "medium": int(medium_count),
                    "hard": int(hard_count),
                },
                seed=int(seed),
                include_answers=include_answers,
                progress_callback=update_progress,
            )
            progress.progress(1.0, text="생성 및 파일 저장 완료")
            st.session_state["dataset_result"] = {
                "zip_name": result["zip_path"].name,
                "zip_data": zip_bytes(result["zip_path"]),
                "run_dir": str(result["run_dir"]),
                "manifest": result["manifest"],
            }
        except Exception as error:
            progress.empty()
            st.exception(error)

if result := st.session_state.get("dataset_result"):
    manifest = result["manifest"]
    st.success("합성 데이터셋과 교육생별 분배 파일을 만들었습니다.")
    first, second, third = st.columns(3)
    first.metric("교육생 파일", manifest["student_count"])
    second.metric("1인당 문항", manifest["questions_per_student"])
    third.metric("전체 고유 문항", manifest["total_unique_questions"])
    st.caption(f"서버 저장 위치: {result['run_dir']}")
    st.download_button(
        "ZIP 파일 다운로드",
        data=result["zip_data"],
        file_name=result["zip_name"],
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
