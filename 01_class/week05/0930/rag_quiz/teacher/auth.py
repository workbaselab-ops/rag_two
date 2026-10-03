"""교사용 패키지의 Streamlit 페이지 인증 기능.

교육생용 챗봇 코드와 교사용 기능을 분리하기 위해 인증 처리는 이 파일에서만
담당한다. 비밀번호는 환경 변수 ``TEACHER_PASSWORD``에서 읽고, 인증 결과만 세션에 저장한다.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

# 강사 원본은 교사용 비밀번호를 이 자리에 문자열로 직접 적었다. Public 저장소(rag_two)에 올리려고
# 값을 지우고 환경 변수 TEACHER_PASSWORD에서 읽도록 바꿨다(희영 허락, 2026-10-03).
AUTH_SESSION_KEY = "teacher_authenticated"


def verify_teacher_password(password: str) -> bool:
    """입력값이 교사용 비밀번호와 일치하는지 안전하게 비교한다."""
    expected = os.getenv("TEACHER_PASSWORD", "")
    if not expected:  # 환경 변수가 없으면 빈 입력으로 통과되지 않도록 항상 거부
        return False
    return hmac.compare_digest(password, expected)


def require_teacher_login() -> bool:
    """교사용 로그인 화면을 표시하고 현재 인증 여부를 반환한다."""
    if st.session_state.get(AUTH_SESSION_KEY, False):
        with st.sidebar:
            st.success("교사용 인증 완료")
            if st.button("교사용 로그아웃", use_container_width=True):
                st.session_state[AUTH_SESSION_KEY] = False
                st.rerun()
        return True

    st.title("🔐 교사용 페이지")
    st.caption("합성 테스트 데이터셋 생성 기능은 교사용 비밀번호가 필요합니다.")

    with st.form("teacher_login_form"):
        password = st.text_input(
            "교사용 비밀번호",
            type="password",
            placeholder="비밀번호 입력",
        )
        submitted = st.form_submit_button(
            "로그인", type="primary", use_container_width=True
        )

    if submitted:
        if verify_teacher_password(password):
            st.session_state[AUTH_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    return False
