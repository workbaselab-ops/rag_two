# 생성형 AI 윤리 RAG 챗봇

`data/생성형_AI윤리_가이드북_RAG시험용_30쪽.pdf`를 참고하는 시험용 챗봇과
교사용 합성 평가 데이터셋 생성 도구입니다.

## 설정

프로젝트 루트의 `.env` 파일에 OpenAI API 키를 설정합니다.

```dotenv
OPENAI_API_KEY=your_api_key
# 선택 사항
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

의존성을 설치합니다.

```bash
uv sync
```

## Streamlit 챗봇 실행

```bash
uv run streamlit run app.py
```

질문과 답변은 세션 동안 유지되며, 각 답변 아래에서 검색된 PDF 근거를 확인할 수 있습니다.

## 교육생 평가 파일 생성

왼쪽 메뉴에서 **교육생 평가파일 생성**을 선택하고 교사가 배포한
`student_XX_questions.json`을 업로드합니다. **평가 실행 및 제출 파일 생성**
버튼을 누르면 교육생의 `rag.ask()`가 각 문항에 답하고 다음 정보를 JSON으로
저장합니다.

- 교육생 ID, 문항 ID와 질문
- RAG 답변, 검색 문맥, 페이지 출처
- 문항별 실행 시간과 성공·오류 상태

교육생 PC에서 답변 생성 직후 RAGAS 평가를 실행하되 산출 점수는 화면에
표시하지 않습니다. `submissions` 폴더에 답변 CSV·JSON, RAGAS 점수 CSV·요약
JSON을 저장한 뒤 하나의 최종 제출 ZIP으로 묶습니다. 교육생은 이 ZIP 또는
`ragas_scores` CSV를 파일 제출 시스템에 업로드합니다.

RAGAS의 Context Precision 계산에 기준 답안이 필요하므로 교사용 페이지에서
새로 생성하는 교육생 문항 파일에는 기준 답안과 근거가 포함됩니다. 따라서 이
방식은 자기평가 편의성을 우선하며 평가 답안의 비공개성은 보장하지 않습니다.

## 합성 테스트 데이터셋 생성

Streamlit 실행 후 왼쪽 페이지 메뉴에서 **교사용 데이터셋 생성**을 선택합니다.
교사용 비밀번호(`.env`의 `TEACHER_PASSWORD`에 넣은 값)를 입력해야 생성 화면을 사용할 수 있습니다. 강사 원본은 비밀번호를 코드에 적었으나 Public 저장소라 환경 변수로 바꿨다(2026-10-03).
교육생 수와 1인당 난이도별 문항 수를 지정한 뒤 버튼을 누르면 다음 파일이
`evaluation/datasets`에 저장되고 ZIP으로도 다운로드할 수 있습니다.

```text
run_YYYYMMDD_HHMMSS/
├── manifest.json
├── teacher/
│   └── master_answer_key.json
└── students/
    ├── student_01_questions.json
    ├── student_02_questions.json
    └── ...
```

- 각 교육생은 난이도별로 동일한 문항 수를 받습니다.
- 한 번의 생성 작업 안에서는 교육생 간 문항 ID가 중복되지 않습니다.
- 교육생의 RAGAS 자기평가를 위해 배포 문항 파일에 기준 답안과 근거가 포함됩니다.
- 기본값은 교육생 23명, 1인당 하 1개·중 1개·상 1개로 총 69문항입니다.
- 합성 문항 생성에는 `OPENAI_API_KEY`가 필요하며 문항 수에 따라 API 비용과 시간이 증가합니다.
- 생성 모델은 `.env`의 `SYNTHETIC_DATASET_MODEL`로 변경할 수 있습니다.

### 교육생용·교사용 파일 구분

```text
app.py                              # 교육생용 챗봇 화면
rag.py                              # 교육생이 개발·연결할 RAG 기능
student/                            # 교육생용 제출 파일 생성 기능
└── submission_service.py           # RAG 실행 결과 구성·JSON 저장
pages/1_교육생_평가파일_생성.py      # 교육생 평가 실행 화면
pages/2_교사용_데이터셋_생성.py      # 교사용 화면 진입점
teacher/                            # 교사용 기능 전용 폴더
├── auth.py                         # 교사용 비밀번호 인증
├── dataset_service.py              # 문항 생성·균등 분배·저장 기능
└── evaluator.py                    # RAGAS 자동평가 실제 구현
evaluator.py                        # 기존 명령을 유지하는 호환 실행 파일
```

비밀번호 인증은 교육용 로컬 실행을 위한 간단한 접근 제한입니다. 공개 서버에서
운영할 때는 고정 비밀번호 대신 환경 변수 또는 별도 인증 시스템을 사용해야 합니다.

### 교사용 기능 테스트

OpenAI API를 호출하지 않는 회귀 테스트로 인증, PDF 로딩, 69문항 생성 흐름,
23명 균등 분배, ZIP 저장 및 평가기 호환성을 확인할 수 있습니다.

```bash
uv run python -m unittest tests.test_teacher_features -v
```

실제 OpenAI 문항 생성과 RAGAS 평가는 별도의 API 호출 테스트가 필요하며 비용이
발생할 수 있습니다.

## FastAPI 실행

```bash
uv run uvicorn api:app --reload
```

- API 문서: <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/health>
- 질의응답: `POST /chat`

요청 예시:

```json
{
  "question": "생성형 AI의 주요 윤리적 위험은 무엇인가요?"
}
```

## 교육생 제출 인터페이스

자동채점과 챗봇 연결을 위해 `rag.py`의 다음 규격을 유지합니다.

```python
def ask(question: str) -> dict:
    return {
        "question": question,
        "answer": "...",
        "contexts": ["검색 문맥 1", "검색 문맥 2"],
        "sources": [{"page": 1, "content": "검색 문맥 1"}],
    }
```

## 강사용 RAGAS 자동채점

`questions.json`은 다음 형식을 사용합니다.

```json
[
  {
    "id": "q001",
    "question": "생성형 AI의 주요 윤리적 위험은 무엇인가요?",
    "reference": "저작권, 개인정보 유출, 허위조작정보 등의 위험이 있습니다.",
    "difficulty": "중",
    "type": "설명"
  }
]
```

질문 파일만 검증합니다.

```bash
uv run python evaluator.py --validate-only
```

강사용 정답 RAG를 평가하고 `scores.csv`를 생성합니다.

```bash
uv run python evaluator.py --student-id instructor
```

교육생이 제출한 CSV 또는 JSON을 다시 실행하지 않고 채점하려면 교사용 마스터
정답지와 제출 파일을 함께 지정합니다.

```bash
uv run python evaluator.py \
  --questions evaluation/datasets/run_YYYYMMDD_HHMMSS/teacher/master_answer_key.json \
  --submission submissions/student_01_submission_YYYYMMDD_HHMMSS.csv \
  --output scores/student_01_scores.csv
```

정답지와 제출 파일의 문항 ID·질문·답변·검색 문맥만 먼저 검사하려면 API 비용이
발생하지 않는 검증 옵션을 사용합니다.

```bash
uv run python evaluator.py \
  --questions evaluation/datasets/run_YYYYMMDD_HHMMSS/teacher/master_answer_key.json \
  --submission submissions/student_01_submission_YYYYMMDD_HHMMSS.csv \
  --validate-only
```

평가 지표와 반영 비율은 Context Precision 30%, Faithfulness 40%, Answer
Relevancy 30%입니다. 실행 결과로 문항별 `scores.csv`와 평균 및 실패 내역이
담긴 `scores_summary.json`이 생성됩니다.

