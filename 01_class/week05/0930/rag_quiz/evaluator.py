"""교사용 자동평가 호환 실행 파일.

기존 ``python evaluator.py`` 명령은 유지하고 실제 구현은 ``teacher/`` 폴더에
분리한다.
"""

from teacher.evaluator import main


if __name__ == "__main__":
    raise SystemExit(main())
