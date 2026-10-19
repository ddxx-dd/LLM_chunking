# AGENTS.md — LLM_chunking

이 저장소에서 일하는 코딩 에이전트(사람 포함)를 위한 작업 지침이다. 프로젝트 배경은 `CLAUDE.md`, 실행 방법은 `README.md`.

## 1. 목표

긴 문서·자막을 어떻게 청킹하느냐가 **검색·생성(요약) 품질**과 **번역 품질**에 어떤 영향을 주는지 같은 조건에서 비교한다.
비교 방식(검색 지표·문서 안 검색 과제)은 LumberChunker(Duarte et al., 2024, `papers/` 참고)를 참고한다.

- 비교 대상: LangChain 기준 청커 3종(fixed, recursive, semantic) + 우리가 만드는 `advanced_semantic_chunker`
- **모든 청커는 같은 깨끗한 텍스트를 입력으로 받는다**(3절 입력 규칙). 청커 밖에서 만든 구조 정보(제목 계층·메타데이터)는 쓰지 않는다. advanced_semantic_chunker는 텍스트 안의 빈 줄·줄바꿈만 쓴다(recursive가 쓰는 것과 같은 정보). 이 효과는 b = 0 절제 실험으로 따로 보고한다.
- `advanced_semantic_chunker`: 문장 임베딩의 코사인 유사도로 **문서 전체에서 청크 안 흩어짐이 가장 작은 자르기를 동적 계획법으로 찾는다**(Embed-KCPD 비용 + 문단 경계 보너스 + 크기 제약). 2026-10-17 LumberChunker 틀(앞에서부터 처음 변화에서 자름)에서 바꿨다.

## 2. 반드시 지킬 것

1. **가상환경 `.venv311` 안에서만** 실행한다. 서버 전역 환경(시스템 파이썬, 전역 pip, 시스템 서비스)은 건드리지 않는다.
2. 새 패키지가 필요하면 **설치 전에 사용자에게 묻는다**(설치는 `.venv311` 안에만).
3. `env.sh`는 `HF_HUB_OFFLINE=1`이다. 모델·데이터를 새로 받을 때만 사용자 허락을 받고 잠시 연다.
4. **실제 데이터에 청킹을 돌리기 전에 크기 값(θ, chunk size, min/max 토큰)을 사용자에게 확인받는다.**
5. 파이프라인 단계마다 **실제 출력을 보여 주고 검수받은 뒤** 다음 단계로 간다(아래 6절). 여러 단계를 한꺼번에 구현하지 않는다.
6. 삭제·덮어쓰기는 대상을 먼저 보여 주고 승인받는다. 옛 코드는 지우기 전에 보존 위치를 확인한다(9절).
7. git: 개발 브랜치는 `v3`. **원격 푸시는 사용자가 한다**(이 환경에는 GitHub 인증이 없다). 강제 푸시 금지.
8. 추정치는 추정이라고 밝히고, 실패·건너뛴 단계는 그대로 보고한다.

## 3. 비교할 청커

| 이름 | 구현 | 크기 조절 |
|---|---|---|
| `fixed` | LangChain `CharacterTextSplitter` | chunk size |
| `recursive` | LangChain `RecursiveCharacterTextSplitter` | chunk size |
| `semantic` | LangChain `SemanticChunker`(`langchain_experimental`) | 경계 백분위 |
| `advanced_semantic_chunker` | 우리 구현 | **target**(평균 청크 토큰, β 이분 탐색) 또는 β, θ(상한), min_tokens(하한) |

- `header`(MarkdownHeaderTextSplitter)는 **제외**한다(2026-10-12): 제목 인식 품질이 고르지 않고(vectara 계층 오류, allganize는 모든 제목이 같은 수준, docx 소제목이 문단으로 들어감) LongBench에는 제목이 없다. 구조 기반 청킹과의 비교가 빠진다는 점은 한계로 적는다.
- LumberChunker처럼 문단 단위 기준선(문단 하나 = 청크 하나)을 추가할지는 A3에서 정한다.

### 입력: clean_text (모든 청커 공통)

전처리는 `src/preprocess/`: 1차 `stage1`이 원본 → Docling 변환(`data/docling_json`) → 블록 + 위치·줄 정보(`data/meta_text_v2`), 2차 `stage2`가 정리 규칙으로 단위를 만든 뒤 `clean_text.build`로 **문서 하나 = 긴 텍스트 하나**인 `data/clean_text/<데이터셋>/<doc_id>.json`을 만든다(2026-10-17 합침; 옛 `processed_para`·`processed_sen`은 지움). 규칙을 바꾸면 Docling은 다시 돌리지 않고 `stage1 --build-only`와 `stage2`만 다시 돌린다.

형식(v5, 2026-10-19): v4(Vectara 공식 코퍼스 = Open RAG Benchmark의 Mistral OCR 텍스트와 비교해 정함, 비교 도구 `tests/clean_text/ref_compare.py`)를 만든 뒤
`src/preprocess/paragraphs.py`가 문단을 의미상 다시 묶는다(사용자 결정: 공식 형식보다 문단을 깔끔하게 나누는 것을 우선). 공식 코퍼스 대비 본문 재현 98.4%, 잡음 2.1%,
순서 98.2%, 문단 시작 재현·정밀 64.3%·79.4%(v4 74.0%·78.9%, 수식·목록을 합쳐서 예상된 하락). 시험 구현·규칙별 검사는 `tests/clean_text/v5.py`, `tests/clean_text/out/v5/`.

| 항목 | 형식 |
|---|---|
| 문단 사이 | 빈 줄 `\n\n` |
| 제목과 본문 사이 | `\n`(마크다운 표시 없음). 단독 제목·장 번호 문단도 다음 본문 앞에 `\n`으로 붙임 |
| 짧은 조각을 붙인 자리, 끊긴 문단을 이은 자리 | 공백(앞 본문이 문장 끝 없이 끝나고 다음 본문이 소문자·`(`로 시작하면 이음, 사이 캡션·각주·표는 이은 문단 뒤로) |
| 원문 줄바꿈(표 행, Docling 블록 안) | 유지 |
| 논문 제목·저자 | 본문 앞 |
| 디스플레이 수식·코드 | `$$ LaTeX $$`를 앞 문장 뒤에 **공백으로** 붙임(문장 안에 들어간 것). 뒤가 이어지는 모양(소문자, where …)이면 뒤 문장도 공백으로 |
| 목록 | 도입 문장(`:` `—` …) + 연속 항목(글자로 쓴 조항 번호 `(a)` `(i)` `(2)` 포함)을 `\n`으로 이은 한 문단. 윗단계 글머리(□ Ⅰ. 1. Part …)에서 새 묶음, 200낱말 상한 |
| 목차 | 쪽 번호로 끝나는 연속 줄을 `\n`으로 이은 한 문단(kind toc) |
| 참고문헌 | 남김. 연속된 항목을 `\n`으로 이은 한 문단 + Docling 목록 표지(`[12]`). 제목 없는 참고문헌도 묶고, pdf 2단 배치로 본문 사이에 끼어든 묶음은 문서 끝으로 |
| 표 | 본문 안 마크다운, 캡션과 한 블록. 참고문헌 영역의 참고문헌 모양 표는 항목 줄로 폄 |
| 그림 캡션 | 남김(원래 위치). 그림 안 글자는 지움 |
| 각주 | 문서 끝 |

파일: `{doc_id, dataset, lang, title, fmt, source, n_pages, n_paras, n_tokens, text, paras[{kind, pages, block_ids}], log}`, 지운 블록은 `<doc_id>.dropped.json`. `paras`는 `text`를 빈 줄로 나눈 문단과 순서대로 대응하고 평가·추적용이다(kind: para/list/toc/caption/footnote/table/heading/references, 수식·코드는 v5에서 문단 안으로 들어감). **청커는 `text`만 쓴다.** 문장이 필요한 청커(semantic, advanced_semantic_chunker)는 같은 공용 문장 함수로 `text`를 나눈다(`src/preprocess/sentences.py`의 함수: 영어 nltk + 학술 약어 + 보정, 한국어 Kiwi + 구조 표시; 청커 쪽 공용 함수는 A3에서 만든다). LongBench(HotpotQA·NarrativeQA 책 14편)는 clean_text를 만들지 않고 `src/preprocess/longbench.py`가 원본 jsonl의 context를 그대로 청커 입력으로, input·answers를 질문·정답으로 읽는다(GovReport는 2026-10-17 삭제).

### `advanced_semantic_chunker` (`src/chunking/advanced_semantic.py`, 2026-10-17)

```
단위      chunking.sentences.split_structured: 문장(표·수식은 한 단위) + 앞 구분(\n\n / \n / 공백) + 제목 여부
점수      Σ_청크 흩어짐 + β·(청크 수) − b·(빈 줄 문단 시작에서 자른 수)          ← 작을수록 좋음
          흩어짐([i, j)) = n·(1 − ‖평균 벡터‖²) = n − ‖Σe‖²/n                  ← Embed-KCPD(arXiv 2601.18788) 코사인 커널 비용
제약      min_tokens ≤ 청크 토큰 ≤ θ (θ보다 큰 단위 하나는 혼자 허용), 제목 줄 바로 뒤에서 자르지 않음
탐색      동적 계획법 best[j] = min_i best[i] + 흩어짐([i, j)) + β − b·[i가 문단 시작]   (i는 θ 안에 드는 범위만)
β         target(평균 청크 토큰)에 맞도록 문서마다 이분 탐색(라벨 없음) 또는 고정값, b = bonus_frac·β (기본 0.5)
청크 글   단위를 원래 앞 구분으로 다시 이음(문단·제목 줄 형식 유지)
```

| 요소 | 결정 |
|---|---|
| 판단 방식 | 문서 전체 최적화. 앞에서부터 처음 변화에서 자르던 옛 판(LumberChunker 틀 + Max–Min / 중심 벡터 μ−0.5σ)은 2026-10-17 파일럿에서 HotpotQA 패시지 경계 재현율 0.27~0.31(새 판 b=0은 0.75), 문서 안 검색도 새 판 이하라서 지웠다(`tests/chunking/out/compare_v2_first_run.md`, 코드는 git 기록) |
| 구조 사용 | 빈 줄 문단 경계 보너스 b와 제목 뒤 절단 금지. **b = 0·제목 규칙 없음 판을 절제 실험으로 함께 보고**한다. HotpotQA는 패시지 사이에 빈 줄이 없어(패시지 안 절 사이에만 있음) 보너스가 패시지 경계 재현율을 0.75 → 0.29로 낮췄다(한계로 적음) |
| 크기 | 같은 조건 비교를 위해 target으로 평균 크기를 기준 청커와 맞춘다. 값(target, θ, min_tokens, bonus_frac)은 개발 문서에서 정하고 실제 실행 전 사용자 확인 |
| 파일럿 임시 값 | θ 550, min_tokens 100, target 200, bonus_frac 0.5 |
| 계산량 | 150편(vectara 50·docx 40·HotpotQA 60)에 약 130초(β 이분 탐색 18회 포함, 단위 임베딩 제외) |

## 4. 파이프라인

### 트랙 A: 검색·생성 (`src/rag/`) — pdf, docx, LongBench

```
원본 → 전처리(Docling → data/meta_text_v2 → data/clean_text: 문서 하나 = 긴 텍스트 하나)
     → advanced_semantic_chunker의 크기·보너스 값 찾기(개발 문서) + 청커별 청킹
     → 임베딩(bge-m3 fp16) → FAISS 정확 검색 → top-k
     → 검색 품질 측정 + top-k를 넣은 생성(질의응답·요약) 품질 측정 → 지표화
```
- 청크는 **문자열**로 다룬다. 오프셋(`start/end`)은 쓰지 않는다.
- 검색 지표 후보: Recall@k, DCG@k(LumberChunker와 같은 정의), MRR. 생성 지표 후보: Gemma 판정, F1/ROUGE 등. **지표는 단계별 검수로 하나씩 확정한다.**

### 트랙 B: 자막 번역 (`src/srt/`) — srt(영화 5편, EN/KO 쌍)

```
srt → 전처리(큐 + 타임스탬프) → 청커별 청킹 → 청크 단위 번역(Gemma)
    → 원래 큐·타임스탬프로 재병합 → EN/KO 시간 정렬 → 큐 단위 번역 품질 평가
```
- **오프셋은 트랙 B에서만 쓴다.** 필요한 곳은 "청크 → 큐 조각" 계산 한 단계다(legacy `mapper.fragments`).
  1. 큐를 이어 붙인 자막 글과 큐별 글자 구간을 만든다(`srt/render.py`).
  2. 청크마다 자막 글 기준 글자 구간 [시작, 끝)을 구한다(`srt/spans.py`): 글자 단위 청커(fixed, recursive)는 LangChain `add_start_index=True`의 시작 위치, 단위(큐) 범위를 돌려주는 청커는 큐 구간을 이어서 계산.
  3. 청크 구간과 겹치는 큐 조각을 찾아 `[1] … [2] …` 번호를 붙여 번역하고, 번호로 다시 나눠 큐별로 합친다(잘린 큐는 앞뒤 조각 번역을 이어 붙인다). 원래 타임스탬프로 SRT를 저장한다.
  4. 참조 KO 큐를 시간 겹침(Jaccard)으로 정렬해 큐 단위로 채점한다.
- 큐가 실제로 잘리는 것은 글자 단위 청커뿐이다. 큐 단위로 묶는 청커(advanced_semantic_chunker, 원본 min/max 엔진)는 경계가 항상 큐 경계라 조각이 큐 전체지만 같은 매핑 코드로 처리한다.
- 원본 알고리즘(min/max 의미 분할 DP, smart)은 **수정하지 않고** 가져다 쓴다. min/max 토큰은 LLM 입력 상한·부하 분산 용도다.
- 옛 코드는 `/root/jupyter/LLM_chunking_legacy/`에서 읽고 `src/`로 옮겨 온다(import하지 않는다): srt 전처리·렌더·원본 알고리즘은 `src/srt/`, `src/chunking/sizectl/`, `src/chunking/v1/`; 번역·매핑·정렬·채점은 `legacy/src/srt/mapper.py`, `subtitle_translate.py`, `timestamp_align.py`, `legacy/src/comet_score.py`.
- 평가 후보: chrF, BLEU, BERTScore(참조 KO 자막), Gemma 판정, CometKiwi(`.venv-eval`, 모델 확보 시).

## 5. 공정한 비교 원칙

- 모든 청커는 같은 깨끗한 텍스트, 같은 임베딩(bge-m3 fp16), 같은 검색기(FAISS 정확 검색), 같은 k, 같은 생성 모델·프롬프트를 쓴다.
- θ는 **개발 문서**에서만 고르고, 평가는 겹치지 않는 문서로 한다. 통계는 문서 단위 부트스트랩.
- 청크 크기가 결과에 영향을 주므로 평균 청크 토큰과 top-k 총 토큰을 항상 함께 보고한다.
- 자체 설계 값(근거 논문이 없는 상수)은 그렇다고 명시한다.

## 6. 개발 방식 (단계별 검수)

각 단계: 작은 코드 작성 → **실제 데이터로 출력 확인(사용자 검수)** → 테스트 추가 → 커밋 → 다음 단계.

| 단계 | 내용 |
|---|---|
| R0 | **완료(2026-10-12)**: 코드·문서·옛 산출물을 모두 지우고 빈 폴더 구조에서 다시 시작. 환경, `AGENTS.md`, `CLAUDE.md`, requirements, `data/raw·processed(현 meta_text)·qa·manifests`만 남김 |
| A1 | 모델 로딩: bge-m3 fp16, 토크나이저, Gemma 4비트 |
| A2 | **완료(2026-10-17)**: 데이터셋 등록, 전처리 v2 이식, 문장 나누기 도구 결정(A2-3), LongBench 단위(A2-4), clean_text v4 형식(stage2에 합침). 남은 것: LongBench clean_text 형식 |
| A3 | 청킹: fixed → recursive → semantic → advanced_semantic_chunker(파일럿) |
| A4 | 인덱싱: 청크 → 임베딩 → FAISS, 캐시 |
| A5 | QA 세트(docx는 Claude 생성 + Gemma 검증) → 지표를 하나씩 추가 |
| A6 | 크기·파라미터 튜닝(개발 문서) → 평가 실행 → 분석 |
| B1–B6 | srt: 점검 → 청킹 → 번역 → 재병합·정렬 → 평가 → 실험 (A4 이후 시작) |
| Z | README(설치·실행·재현), 지표 정의서, 트러블슈팅, 한계 |

## 7. 폴더 구조 (목표, `v3`)

```
src/
  common/                       설정, 모델 로딩(bge-m3·Gemma), 공용 텍스트 함수(문장 나누기 등)
  preprocess/                   전처리: sources(문서 목록), docling_blocks(Docling 변환·블록), stage1(1차), stage2(2차 → clean_text),
                                clean_text(clean_text 형식 정리), sentences(문장 나누기 함수), longbench(LongBench 원본 읽기)

  chunking/                     ── 청커 (오프셋을 모른다). 청커마다 파일 하나, 입력은 문서 텍스트(pdf·docx는 clean_text의 text, LongBench는 원본 context)
    sentences.py                  공용 문장 분리(semantic·advanced_semantic만): 빈 줄 = 문단, 표·수식은 한 단위, 줄(접힌 줄은 이음) → preprocess.sentences.split
    fixed.py                      LangChain CharacterTextSplitter, 글자 수(기본 separator='' = 순수 고정 길이)
    recursive.py                  LangChain RecursiveCharacterTextSplitter 기본 그대로(글자 수)
    semantic.py                   LangChain SemanticChunker 기본 그대로, 문장 분리만 sentences.split_text
    advanced_semantic.py          문서 전체 동적 계획법(KCPD 코사인 비용 + 문단 보너스 + θ·min 제약 + 제목 규칙), β는 target 크기로 이분 탐색, 크기는 bge-m3 토큰
    embeddings.py                 bge-m3 encode, LangChain Embeddings 감싸기
    sizectl/                      srt 원본 min/max 엔진 → 단위 번호 범위 (옛 코드 이식, 수정 없이)

  rag/                          ── 트랙 A: 오프셋 없음
    chunks.py                     단위 범위 → 청크 문자열 (LangChain 결과는 문자열 그대로)
    index.py                      청크 → 임베딩 → FAISS 정확 검색, 캐시
    metrics.py                    검색 지표 (단계별로 하나씩 추가)
    judge.py                      Gemma 판정, 생성(질의응답·요약) 품질
    tune.py                       θ 격자 튜닝, 문서 단위 부트스트랩
    datasets/                     docx, vectara, allganize, LongBench 로더

  srt/                          ── 트랙 B: 오프셋은 여기에만
    prep.py, validate.py          srt → 큐 JSON (옛 코드 이식)
    render.py                     큐 → 자막 글 + 큐별 글자 구간 (옛 코드 이식)
    spans.py                      어떤 청커 결과든 → 청크의 글자 구간 [시작, 끝)
    mapper.py                     청크 구간 → 큐 조각 → 번호 프롬프트 → 번역 분리 → 큐별 병합 → SRT 저장 (legacy 이식)
    translate.py                  Gemma 청크 번역, 배치 (legacy 이식)
    align.py                      EN/KO 시간 겹침 정렬 (legacy 이식)
    eval.py                       chrF/BLEU/BERTScore, 타임스탬프 무결성, CometKiwi(.venv-eval) (legacy 이식)

  qa_build/                     docx QA 구축 도구

scripts/        실행 CLI (preprocess, build_qa, chunk, run, tune, srt/…)
tests/          unittest(모델·GPU 없이 돈다) + 검증 도구: clean_text/ref_compare.py(Vectara 정답 비교), sent_split/(문장 도구 판정 자료)
data/           raw → docling_json → meta_text_v2 → clean_text(청커 입력) → chunks → index (git에는 qa·manifests만; LongBench는 raw를 그대로 읽음)
docs/ papers/   설계 문서 / 참고 논문 PDF(git에 올리지 않음)
```

청크 기록의 형태:

| 트랙 | 청크 기록 |
|---|---|
| rag | `{chunk_id, doc_id, text, n_tokens}` |
| srt | `{chunk_id, movie, text, start, end, cue_from, cue_to}` — `start/end`는 렌더된 자막 글 기준 글자 위치 |

- LangChain 청커는 모두 `add_start_index`로 시작 위치를 받을 수 있다(`SemanticChunker`도 지원, `langchain_experimental` 0.4.2에서 확인). 트랙 B에서만 쓴다.
지금은 폴더와 빈 `__init__.py`만 있다. 파일은 단계마다 하나씩 추가하고, 추가할 때 이 절을 실제 구조에 맞춘다.

## 8. 실행·테스트

```bash
source env.sh                                   # HF 캐시·PYTHONPATH 고정, 오프라인
.venv311/bin/python -W ignore -m unittest discover -s tests   # pytest는 설치되어 있지 않다
```
- GPU: RTX A5000 24GB. Gemma 4 12B는 `transformers` + `bitsandbytes` 4비트로 약 7.7GiB.
- 오래 걸리는 작업은 로그 파일로 남기고 백그라운드로 돌린다. CUDA 정리 단계에서 프로세스가 오래 매달린 적이 있어 GPU 스크립트는 끝에 `os._exit(0)`을 쓴다(이때 출력 버퍼가 비워지지 않으므로 결과는 파일로 저장).
- CometKiwi는 `transformers` 버전 충돌 때문에 별도 환경 `.venv-eval`에서만 돌린다. **`.venv-eval`은 현재 실행되지 않는다**(2026-10-12 확인): `bin/python`이 이전 서버 경로(`/root/.local/share/uv/python/...`)를 가리킨다. 패키지(`unbabel-comet` 2.2.7 등)는 남아 있다. B5에서 사용자 확인 후 복구한다(파이썬 연결을 `.python/cpython-3.11.17`로 바꾸거나 `requirements-eval.txt`로 재생성). CometKiwi 모델도 아직 캐시에 없다.
- **GPU 합성곱(cuDNN) 불가**(2026-10-12 확인): torch 2.7.1+cu118이 요구하는 `nvidia-cudnn-cu11` 9.1과, `cuda-toolkit 13` 묶음으로 들어온 `nvidia-cudnn-cu13` 9.24가 **같은 파일**(`nvidia/cudnn/lib/libcudnn.so.9`)에 설치되어 9.24가 덮어썼다. 작은 `conv2d`도 `CUDNN_STATUS_NOT_INITIALIZED`로 실패한다. bge-m3·Gemma는 영향 없음. Docling pdf(레이아웃·표·OCR 모델)는 GPU에서 실패하고, CPU 변환 시도도 오류 없이 중단됐다(원인 미확인). docx 변환은 정상. A2-5에서 사용자 허락 후 `nvidia-cudnn-cu11==9.1.0.70` 재설치로 수리한다(CUDA 13 묶음을 쓰는 패키지 영향 확인 후).
- `.venv311` 점검(2026-10-12): 주요 패키지 import, CUDA(A5000, cu118), 오프라인 bge-m3 fp16 인코딩, Gemma 토크나이저, Docling 모델 캐시 정상. `pip check` 경고 4건(`unstructured`·`pdfplumber`·`pi-heif`·`nab-project`의 버전 요구)은 우리가 쓰지 않는 패키지라 그대로 둔다.

## 9. 옛 코드와 보존 위치

- 태그 `exp1-offset-chunking`(원격 브랜치 `v2-restructure`와 같은 커밋 `07a8722`): 오프셋 기반 청커 7종(A/B/E 포함), 실험 1 파이프라인.
- `/root/jupyter/LLM_chunking_legacy/`: 위 시점의 코드 전체 + 추적되지 않던 산출물(`results`, `data/chunks`, `data/index`) + 당시 패키지 목록(`requirements-freeze-venv311.txt`). 그 안의 `legacy/`는 구조 개편 이전 코드(srt 번역·재병합·시간 정렬·CometKiwi).
- `data/raw`는 git과 legacy 폴더 어디에도 사본이 없다. **삭제 금지.**
- 원격 `v2` 브랜치는 로컬과 갈라져 있다(srt 다운로더 등 4개 커밋). srt 트랙 시작 때 병합을 검토한다.

## 10. 코드 규칙

- 주변 코드의 문체를 따른다: 한국어 docstring·주석, 짧은 순수 함수, 설정값은 `common/config.py`.
- 새 기능에는 모델·GPU 없이 도는 단위 테스트를 붙인다(가짜 임베딩 사용).
- 데이터: `data/qa`, `data/manifests`만 git 추적. `papers/`의 PDF는 커밋하지 않는다.

## 11. 미결정 사항

- advanced_semantic_chunker의 target·θ·min_tokens·bonus_frac(개발 문서), 문단 보너스 자리(빈 줄 vs 제목 줄 앞)
- 청크 초반 오판으로 생기는 작은 청크 대책(필요할 때)
- 기준 청커의 크기 값(공통 기본값으로 둘지, 같은 튜닝 예산을 줄지), 문단 단위 기준선 추가 여부(A3)
- 제목처럼 보이는 짧은 줄까지 붙일지(2안, A2-2 통계 후), 문장 나누기 도구(A2-3 검증 후)
- pdf 재변환 대조와 cuDNN 수리(A2-5, 가상환경 변경 허락 필요)
- 검색·생성(요약) 지표 집합과 k
- LongBench 평가 방식(샘플별 검색 / 합친 인덱스), 다운로드 허용
- docx QA v2의 창 길이·문서당 질문 수·근거 형식
- srt 번역 평가에서 CometKiwi 사용 가능 여부
