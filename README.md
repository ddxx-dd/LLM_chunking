# LLM Chunking Pipeline

DIP Challenge Lab 프로젝트 2-1 - fixed/semantic/smart 세 청킹 전략이 검색·요약·번역
품질과 원본 구조 병합에 어떤 영향을 주는지 비교하는 파이프라인. 설계는 `smart_chunk.md` 참고.

## 설치 (venv 2개)

번역 실행(`.venv311`)과 CometKiwi 채점(`.venv-eval`)은 `transformers` 버전이 서로
충돌해서 같은 프로세스에서 못 돌린다(`smart_chunk.md` 8절) - 반드시 두 venv를 분리해서
쓴다.

```bash
# 1) .venv311 - 파싱/청킹/검색/요약/번역/병합 전부(CometKiwi 채점만 빼고)
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
deactivate

# 2) .venv-eval - CometKiwi 채점 전용
python3 -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
deactivate
```

## 실행 순서

모든 스크립트는 `src/` 안에서 `python3 -m <모듈>` 형태로 실행한다(패키지 임포트가
`src/`가 sys.path 루트라고 가정함).

```bash
source .venv311/bin/activate
cd src

# 1. 파싱(문서당 1회, data/processed/에 JSON 캐싱 + data/processed/anchored/에 anchored docx)
python3 -m docx_track.parse <docx 경로>
python3 -m pdf_track.parse <pdf 경로>

# 2. 항등 테스트 + 표시 테스트(원문 그대로 되돌려 써서 위치 정확도 검증, results/checks/에 출력)
python3 -m docx_track.identity_test <docx 경로>
python3 -m pdf_track.identity_test <pdf 경로>
python3 -m merge_checks <docx 경로>   # 표시 테스트 + 서식 섞인 문단 수

# 이후 단계(청킹/검색/요약/번역/병합)는 smart_chunk.md 10절 구현 순서 참고
```

CometKiwi 번역 채점만 별도 venv로 전환해서 실행한다:
```bash
# .venv311에서 번역 실행 -> 결과 파일 저장 -> 프로세스 종료
source .venv-eval/bin/activate
python3 -m evals.comet_score <저장된 번역 결과>
```

## 폴더 구조

`smart_chunk.md` 9절 참고. 원본 데이터(`data/allganize/`, `data/vectara_ragbench/` 등)
에는 아무것도 쓰지 않는다 - 중간 산출물(anchored docx, elements JSON)은
`data/processed/`, 항등/표시 테스트 결과물은 `results/checks/`에 모은다.
