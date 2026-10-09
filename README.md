# LLM Chunking Pipeline

fixed(글자수 분할) · recursive · header(LangChain 구조 기반) · semantic(단순 의미 분할)과 우리가 만든 구조+의미 청커(A, B, E)가
검색 품질에 미치는 영향을 같은 조건에서 비교하는 실험 파이프라인. 대용량 비정형 문서(pdf·docx)와 자막(srt)을 의미 단위로
분할(split)하고 평가한다. 설계 배경과 한계는 [`docs/design.md`](docs/design.md), 구현 계획은
[`docs/restructure_plan.md`](docs/restructure_plan.md), 2026-10-07 폴더 정리 내역은 [`docs/moves_2026-10-07.md`](docs/moves_2026-10-07.md).

## 파이프라인 한눈에

```
원본(data/raw) → ① 전처리 → 공통 JSON(data/processed) → ② 청킹 → 청크 파일(data/chunks) → ③ 인덱스(data/index) → ④ 검색·평가 → results/
                                      ↑                                                                     ↑
                                 문서 하나 = JSON 하나(blocks 목록)                              QA 파일(data/qa): 질문·정답·근거
```

**① 전처리는 형식마다 방식이 다르고, 결과 JSON의 모양(`blocks` 목록)만 같다.**

| 형식 | 전처리 | 블록 종류 | 코드 |
|---|---|---|---|
| pdf, docx | **Docling**(구조 인식) | heading(수준), paragraph, table, list_item, formula, footnote … | `src/preprocess/` (`pdf_prep.py`, `docx_prep.py`) |
| srt(자막) | **정규식 파서**(Docling을 쓰지 않음, `parser: regex-srt`). 인코딩을 순서대로 시도해 읽고 큐(번호·시각·대사) 추출, 태그 제거·정규화, 빈 큐 제외 | cue(시작·끝 시각 포함). 제목·표 같은 구조 정보 없음 | `src/srt/prep.py` |

**② 청킹**은 JSON의 블록을 이어 **평문 하나**(`common/docjson.render`, srt는 `srt/render.py`)로 만들고, 청커가 평문의 연속 구간 `[(start, end)]`을 돌려준다.
pdf·docx는 새 청커 7종(`src/chunking/`), srt는 옛 경로(`chunking/v1` + `chunking/sizectl` + `srt/smart.py`)를 쓴다.
**③ 인덱스**는 청크를 bge-m3 fp16으로 임베딩해 FAISS에 넣고, **④ 평가**는 질문마다 상위 3개 청크를 찾아 지표로 채점한다.

## 설치

모든 작업은 가상환경 `.venv311` 안에서만 한다(서버 전역 환경을 바꾸지 않는다).

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt          # Docling, sentence-transformers, FAISS, LangChain 등
# 평가 전용(CometKiwi) 환경은 별도: python3.11 -m venv .venv-eval && pip install -r requirements-eval.txt
source env.sh                            # HF 캐시 경로 고정, 오프라인 모델 사용
```

## 실행 순서 (pdf·docx)

```bash
source env.sh
PY=.venv311/bin/python
$PY scripts/preprocess.py --dataset vectara            # 원본 -> 공통 JSON (data/processed/)
$PY scripts/measure_ratio.py --dataset all             # 글자/토큰 비율 (글자수 청커의 T 환산용)
$PY scripts/chunk.py --dataset vectara --chunker A --p 90        # 청크 파일 (data/chunks/vectara/A__p90.jsonl)
$PY scripts/chunk.py --dataset vectara --chunker fixed --T 260
$PY scripts/run.py --task retrieval --dataset vectara --tag A__p90,fixed__T260            # 인덱스(FAISS) + 검색 평가
$PY scripts/run.py --task retrieval --datasets vectara,docx_corpus,allganize --tag A__p90  # 한 인덱스에 합쳐서 평가
$PY scripts/chunk_stats.py --dataset vectara --tag A__p90      # 청크 크기·구조 훼손 통계
$PY -m unittest discover -s tests                              # 테스트(모델·GPU 불필요)
```

- 청커 7종(`fixed, recursive, header, semantic, A, B, E`)과 크기 조절 값(`--p` 백분위, `--T` 목표 토큰)은 [`src/chunking/README.md`](src/chunking/README.md).
  **크기 값에는 기본값이 없다** - 실행할 때 명시한다.
- 임베딩은 bge-m3 fp16으로 고정(fp32 대비 top-3 99% 일치, 약 4배 빠름). 인덱스는 `data/index/<dataset>/<태그>@fp16`에 캐시된다.
- 청크 크기가 달라지면 평가 지표가 크게 달라진다(작은 청크는 F1·section_hit에, 큰 청크는 evidence_recall에 유리). 비교할 때는 크기 분포(`chunk_stats.py`)를 같이 본다.
- 결과: `results/<dataset>/retrieval/<태그>.json`(합침 모드는 `results/combined/<dataset>/`).

## srt(자막)

```bash
$PY scripts/preprocess.py --dataset srt
$PY scripts/srt/chunk.py --chunker semantic --percentile 90 --min-tokens 128 --max-tokens 512
```
srt는 Docling이 아니라 **정규식 파서**로 전처리한다(`src/srt/prep.py`). 렌더·검증·smart 청커·설정도 `src/srt/` 패키지에 있고,
청킹은 min/max 엔진(`src/chunking/sizectl/`)을 쓴다 → [`src/srt/README.md`](src/srt/README.md).
srt에는 제목·표 구조가 없어서 구조 규칙을 쓰는 새 청커(A, B, E)의 대상이 아니다.

## 폴더 구조

```
src/
  chunking/        pdf·docx 청커 7종 (base, units, embed, textual, semantic, smart_a/b/e, io)
    sizectl/       min/max 크기 제어 의미 분할 엔진 (srt가 사용, 나중에 7종에 재사용)
    v1/            실험 1(옛 pdf·docx 파이프라인) 보존
  srt/             자막 전용: prep(정규식 파서), render, validate, smart, config
  common/          경로·데이터셋 설정, 공통 JSON 렌더(docjson), 모델·토크나이저
  preprocess/      pdf·docx 전처리(Docling), 전처리 대상 목록, 검증
  indexing/        FAISS 인덱스(정확 검색) 구축·캐시·합치기
  tasks/           검색 평가 지표
  qa_build/        QA 생성·필터(docx_corpus, vectara)
scripts/           CLI (preprocess, measure_ratio, chunk, run, chunk_stats, build_qa, fetch_vectara ...)
  srt/             srt 청킹 CLI
  legacy/          실험 1(옛 API) 전용 스크립트 - 실험 1 재현용, 자세한 것은 scripts/legacy/README.md
tests/             test_chunking/(새 청커) test_v1/ test_sizectl/ test_srt/ + 공용 테스트(unittest)
docs/              설계·계획 문서, 이동 내역(moves_2026-10-07.md), history/(지난 보고서, 현재 코드와 안 맞을 수 있음)
legacy/            구조 개편 이전(루트 src) 코드 보관
data/, results/    입력·중간 산출물·실행 결과(git 추적 안 함. data/qa, data/manifests만 추적)
```
