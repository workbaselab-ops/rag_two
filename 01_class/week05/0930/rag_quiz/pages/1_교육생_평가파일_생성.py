"""교육생이 자신의 RAG 실행 결과를 제출용 파일로 만드는 화면."""

from __future__ import annotations

import streamlit as st

from rag import ask
from student.ragas_scoring import (
    create_submission_package,
    evaluate_submission,
    save_score_files,
)
from student.submission_service import (
    build_submission,
    load_assigned_questions,
    save_submission_files,
)

st.set_page_config(page_title="교육생 평가 파일", page_icon="📤", layout="centered")

st.title("📤 교육생 평가 파일 생성")
st.caption("배정 문항으로 RAG를 실행하고 RAGAS 점수까지 계산해 제출 ZIP을 만듭니다.")
st.info("답변 생성과 RAGAS 평가 모두 OpenAI API를 사용하므로 완료까지 시간이 걸릴 수 있습니다.")

uploaded_file = st.file_uploader(
    "교사가 배포한 문항 JSON 파일",
    type=["json"],
    help="student_XX_questions.json 파일을 선택하세요.",
)

questions: list[dict[str, str]] = []
file_student_id = ""
if uploaded_file is not None:
    try:
        file_student_id, questions = load_assigned_questions(uploaded_file.getvalue())
        st.success(f"문항 {len(questions)}개를 불러왔습니다.")
        with st.expander("평가 문항 확인"):
            for index, item in enumerate(questions, start=1):
                st.write(f"{index}. [{item['id']}] {item['question']}")
    except (UnicodeDecodeError, ValueError, KeyError) as error:
        st.error(str(error))

student_id = st.text_input(
    "교육생 ID",
    value=file_student_id,
    placeholder="예: student_01 또는 사번",
)

if st.button(
    "평가 실행 및 제출 파일 생성",
    type="primary",
    use_container_width=True,
    disabled=not questions,
):
    # 이전 실행 결과가 새 실행 화면에 남지 않도록 먼저 제거한다.
    st.session_state.pop("student_submission", None)
    progress = st.progress(0, text="평가를 준비합니다.")

    def update_progress(completed: int, total: int, question_id: str) -> None:
        progress.progress(
            (completed / total) * 0.65,
            text=f"문항 {question_id} 실행 중 ({completed}/{total})",
        )

    try:
        submission = build_submission(
            student_id,
            questions,
            ask,
            progress_callback=update_progress,
        )
        json_path, csv_path = save_submission_files(submission)
        progress.progress(0.7, text="RAGAS 지표를 계산하고 있습니다.")
        scores, failures = evaluate_submission(questions, submission)
        score_path, summary_path = save_score_files(
            scores, failures, submission["student_id"]
        )
        package_path = create_submission_package(
            submission["student_id"],
            [csv_path, json_path, score_path, summary_path],
        )
        st.session_state["student_submission"] = {
            "json_path": str(json_path),
            "json_filename": json_path.name,
            "json_data": json_path.read_bytes(),
            "csv_path": str(csv_path),
            "csv_filename": csv_path.name,
            "csv_data": csv_path.read_bytes(),
            "score_path": str(score_path),
            "score_filename": score_path.name,
            "score_data": score_path.read_bytes(),
            "summary_path": str(summary_path),
            "package_path": str(package_path),
            "package_filename": package_path.name,
            "package_data": package_path.read_bytes(),
            "success_count": submission["success_count"],
            "error_count": submission["error_count"],
        }
        progress.progress(1.0, text="제출 파일 생성 완료")
    except Exception as error:
        progress.empty()
        st.error(f"평가 파일 생성에 실패했습니다: {error}")

if saved := st.session_state.get("student_submission"):
    if saved["error_count"]:
        st.warning(
            f"성공 {saved['success_count']}개, 실패 {saved['error_count']}개입니다. "
            "파일의 error 항목을 확인하고 필요하면 다시 실행하세요."
        )
    else:
        st.success(f"전체 {saved['success_count']}개 문항 실행에 성공했습니다.")
    # 교육생 화면에는 산출 점수를 노출하지 않는다. 점수는 제출 파일에만 기록한다.
    st.info("평가가 완료되었습니다. 결과 점수는 화면에 표시되지 않으며 제출 파일에 저장됩니다.")
    st.caption(
        f"최종 제출 ZIP: {saved['package_path']}"
    )
    st.download_button(
        "RAGAS 점수 포함 최종 ZIP 다운로드",
        data=saved["package_data"],
        file_name=saved["package_filename"],
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
    with st.expander("개별 결과 파일 다운로드"):
        st.download_button(
            "RAGAS 점수 CSV",
            data=saved["score_data"],
            file_name=saved["score_filename"],
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "RAG 답변 CSV",
            data=saved["csv_data"],
            file_name=saved["csv_filename"],
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "JSON 백업 파일",
            data=saved["json_data"],
            file_name=saved["json_filename"],
            mime="application/json",
            use_container_width=True,
        )
