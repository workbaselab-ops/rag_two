"""교육생용 UI: rag.py를 연결해 사용하는 Streamlit 챗봇.

교사용 합성 데이터셋 생성 기능과 인증은 ``pages/``와 ``teacher/``로
분리되어 있다.
"""

from __future__ import annotations

import streamlit as st

from rag import PDF_PATH, ask, get_status

st.set_page_config(page_title="RAG 챗봇", page_icon="💬", layout="centered")


def _initial_message() -> dict:
    return {
        "role": "assistant",
        "content": (
            "안녕하세요. **생성형 AI 윤리 가이드북**을 바탕으로 답변합니다. "
            "문서 내용에 관해 질문해 주세요."
        ),
        "sources": [],
    }


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"검색 근거 {len(sources)}개 보기"):
        for index, source in enumerate(sources, start=1):
            page = source.get("page", "-")
            st.markdown(f"**근거 {index} · PDF {page}쪽**")
            st.caption(source.get("content", ""))


if "messages" not in st.session_state:
    st.session_state.messages = [_initial_message()]

st.title("💬 생성형 AI 윤리 RAG 챗봇")
st.caption("검색된 PDF 문맥에 근거하여 답변합니다.")

with st.sidebar:
    st.header("챗봇 정보")
    st.write(f"문서: `{PDF_PATH.name}`")
    status = get_status()
    st.write("문서 상태:", "✅ 준비됨" if status["pdf_ready"] else "❌ 없음")
    st.write(
        "API 키:",
        "✅ 설정됨" if status["api_key_configured"] else "❌ 설정 필요",
    )
    if st.button("대화 내용 지우기", use_container_width=True):
        st.session_state.messages = [_initial_message()]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_sources(message.get("sources", []))

if question := st.chat_input("가이드북에 관해 질문하세요"):
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 답을 찾고 있습니다..."):
            try:
                result = ask(question)
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                _render_sources(sources)
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                answer = f"⚠️ {error}"
                sources = []
                st.error(str(error))
            except Exception:
                answer = "⚠️ 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
