# AGENTS.md — LLM_chunking

이 저장소에서 일하는 코딩 에이전트(사람 포함)를 위한 작업 지침이다. 프로젝트 배경은 `CLAUDE.md`, 실행 방법은 `README.md`.

## 1. 목표

긴 문서·자막을 어떻게 청킹하느냐가 **검색·생성(요약) 품질**과 **번역 품질**에 어떤 영향을 주는지 같은 조건에서 비교한다.
비교 방식은 LumberChunker(Duarte et al., 2024, `papers/` 참고)를 따른다.

- 비교 대상: LangChain 기준 청커 3종(fixed, recursive, semantic) + 우리가 만드는 `advanced_semantic_chunker`
- **모든 청커는 같은 깨끗한 텍스트를 입력으로 받는다**(3절 입력 규칙). 청커별 구조 규칙은 쓰지 않고 순수 분할 알고리즘만 비교한다.
- `advanced_semantic_chunker`: LumberChunker의 틀을 그대로 쓰되, **"어디서 내용이 바뀌는가"를 LLM이 판단하던 부분을 코사인 유사도 기반 알고리즘으로 대체**한다.

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
| `advanced_semantic_chunker` | 우리 구현 | **θ**(그룹 토큰 상한, LumberChunker의 θ) |

- `header`(MarkdownHeaderTextSplitter)는 **제외**한다(2026-10-12): 제목 인식 품질이 고르지 않고(vectara 계층 오류, allganize는 모든 제목이 같은 수준, docx 소제목이 문단으로 들어감) LongBench에는 제목이 없다. 구조 기반 청킹과의 비교가 빠진다는 점은 한계로 적는다.
- LumberChunker처럼 문단 단위 기준선(문단 하나 = 청크 하나)을 추가할지는 A3에서 정한다.

### 입력: clean_text (모든 청커 공통)

전처리는 `src/preprocess/`: 1차 `stage1`이 원본 → Docling 변환(`data/docling_json`) → 블록 + 위치·줄 정보(`data/meta_text_v2`), 2차 `stage2`가 정리 규칙으로 단위를 만든 뒤 `clean_text.build`로 **문서 하나 = 긴 텍스트 하나**인 `data/clean_text/<데이터셋>/<doc_id>.json`을 만든다(2026-10-17 합침; 옛 `processed_para`·`processed_sen`은 지움). 규칙을 바꾸면 Docling은 다시 돌리지 않고 `stage1 --build-only`와 `stage2`만 다시 돌린다.

형식(v4, Vectara 공식 코퍼스 = Open RAG Benchmark의 Mistral OCR 텍스트와 비교해 정함; 비교 도구 `tests/clean_text/ref_compare.py`: 본문 재현 98.4%, 잡음 2.0%, 순서 98.2%):

| 항목 | 형식 |
|---|---|
| 문단(블록) 사이 | 빈 줄 `\n\n` |
| 제목과 본문 사이 | `\n`(마크다운 표시 없음, 제목은 다음 블록 앞에 붙임) |
| 짧은 조각을 붙인 자리 | 공백 |
| 원문 줄바꿈(표 행, Docling 블록 안) | 유지 |
| 논문 제목·저자 | 본문 앞 |
| 참고문헌 | 남김. 연속된 항목을 `\n`으로 이은 한 문단 + Docling 목록 표지(`[12]`) |
| 디스플레이 수식 | `$$ LaTeX $$`, 자기 문단(Q3 고장 난 수식은 원본 글자 층) |
| 표 | 본문 안 마크다운, 캡션과 한 블록. 참고문헌 영역의 참고문헌 모양 표는 항목 줄로 폄 |
| 그림 캡션 | 남김(원래 위치). 그림 안 글자는 지움 |
| 각주 | 문서 끝 |

파일: `{doc_id, dataset, lang, title, fmt, source, n_pages, n_paras, n_tokens, text, paras[{kind, pages, block_ids}], log}`, 지운 블록은 `<doc_id>.dropped.json`. `paras`는 `text`를 빈 줄로 나눈 문단과 순서대로 대응하고 평가·추적용이다. **청커는 `text`만 쓴다.** 문장이 필요한 청커(semantic, advanced_semantic_chunker)는 같은 공용 문장 함수로 `text`를 나눈다(`src/preprocess/sentences.py`의 함수: 영어 nltk + 학술 약어 + 보정, 한국어 Kiwi + 구조 표시; 청커 쪽 공용 함수는 A3에서 만든다). LongBench(HotpotQA·NarrativeQA 책 14편)는 clean_text를 만들지 않고 `src/preprocess/longbench.py`가 원본 jsonl의 context를 그대로 청커 입력으로, input·answers를 질문·정답으로 읽는다(GovReport는 2026-10-17 삭제).

### LumberChunker 원본 동작 (논문 3.1절, 그림 1, 부록 B 프롬프트)

1. 문서를 문단 단위로 나누고 번호(ID)를 붙인다.
2. 시작 문단부터 이어 붙여 토큰 수가 θ를 넘는 순간까지 그룹 Gᵢ를 만든다(θ를 넘게 만든 문단 포함).
3. LLM에게 묻는다: "**첫 문단을 제외하고**, 내용이 **앞 문단들과 비교해** 분명히 바뀌는 **첫 번째** 문단의 ID". 프롬프트는 그룹을 너무 길게 만들지 말라고도 지시한다.
4. 그 ID 앞까지가 청크, 그 ID가 다음 그룹의 시작. 문서 끝까지 반복한다.
- θ ∈ [450, 1000] 중 550이 가장 좋았고, 이때 평균 청크는 334토큰(θ보다 약 40% 작음, 표 9)이었다.

### `advanced_semantic_chunker`: LLM 판단만 코사인 유사도 알고리즘으로 대체

```
s = 문서 시작 단위
반복:
  G = s부터 단위를 더해 토큰 합이 θ를 넘을 때까지                     ← 그룹 길이는 θ로 통제
  for j = s+1 … G의 끝:                                              ← 첫 단위는 후보에서 제외
      sim_j = cos( 현재 청크 [s, j)의 중심 벡터 , 단위 j )             ← 바로 앞 하나가 아니라 앞 단위 전체와 비교
      if '분명한 변화' 기준을 만족: c = j; break                       ← 처음 변화가 나타나는 곳에서 자름
  변화가 없으면: c = θ를 넘게 만든 단위 (그 앞까지가 청크, 그 단위부터 재시작)
  청크 = [s, c),  s = c                                              ← 자른 곳에서 다시 시작
```

| 요소 | 결정 |
|---|---|
| 단위 | **문단 또는 문장(옵션 `unit`)**. 어느 쪽이 나은지는 파일럿에서 비교해 정한다 |
| 비교 대상 | 현재 청크 전체의 중심 벡터(토큰 가중 평균) |
| 자르는 곳 | 기준을 처음 만족하는 단위 |
| 그룹 길이 | θ로 통제. 변화가 없으면 그룹 끝(θ를 넘게 만든 단위 앞)에서 절단 → 청크는 θ를 넘지 않는다(θ보다 큰 단일 문단은 문장으로 나눈다) |
| '분명한 변화' 기준 | 세 가지를 구현해 파일럿에서 고른다(2026-10-17): `advanced_semantic` = Max–Min(Kiss et al. 2025, 새 단위의 청크 내 최대 유사도 > max(c·청크 내 최소 쌍 유사도·sigmoid(크기), hard_thr)), `advanced_semantic1` = 중심 벡터 유사도 < 청크 응집도 평균 μ_C(자기를 뺀 중심과의 유사도), `advanced_semantic2` = 중심 벡터 유사도 < μ_C − k·σ_C(k=0.5, TextTiling). 단위 하나일 때는 문서 이웃 유사도 분포에 같은 식. 값은 정답 경계(HotpotQA 패시지, vectara 절)로 고른다 |
| 청크 초반 오판으로 생기는 아주 작은 청크 | 나중에 다룬다(파일럿에서 실제로 문제인지 먼저 확인) |

- θ는 개발용 문서에서 격자로 찾는다. 처음 변화에서 자르므로 청크 크기는 θ보다 '분명한 변화' 기준이 더 크게 좌우하고, θ는 사실상 상한이다.
- 단위(`unit`): 위 '깨끗한 텍스트'의 문단 또는 문장. θ보다 큰 문단은 문단 단위에서도 문장으로 나눈다(표는 예외로 그대로 한 단위). 구조 규칙(제목 가산점 등)은 쓰지 않는다.
- 검증 후보: 같은 θ로 원본 LumberChunker(Gemma, 논문 프롬프트)를 몇 편 돌려 절단 위치 일치도를 재는 방법이 있다(선택).

## 4. 파이프라인

### 트랙 A: 검색·생성 (`src/rag/`) — pdf, docx, LongBench

```
원본 → 전처리(Docling → data/meta_text_v2 → data/clean_text: 문서 하나 = 긴 텍스트 하나)
     → advanced_semantic_chunker의 최적 θ 찾기(개발 문서) + 청커별 청킹
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
| A6 | θ 튜닝 → 평가 실행 → 분석 |
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
    advanced_semantic.py          LumberChunker 틀 + 판단식 3종(advanced_semantic = Max–Min, advanced_semantic1·2 = 중심 벡터 후보 1·2) → 단위 범위 [(i, j)], θ는 bge-m3 토큰
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

- advanced_semantic_chunker의 '분명한 변화' 기준(논문 조사와 함께 개발)과 θ 격자
- advanced_semantic_chunker의 단위: 문단 vs 문장(파일럿 비교)
- 청크 초반 오판으로 생기는 작은 청크 대책(필요할 때)
- 기준 청커의 크기 값(공통 기본값으로 둘지, 같은 튜닝 예산을 줄지), 문단 단위 기준선 추가 여부(A3)
- 제목처럼 보이는 짧은 줄까지 붙일지(2안, A2-2 통계 후), 문장 나누기 도구(A2-3 검증 후)
- pdf 재변환 대조와 cuDNN 수리(A2-5, 가상환경 변경 허락 필요)
- 검색·생성(요약) 지표 집합과 k
- LongBench 평가 방식(샘플별 검색 / 합친 인덱스), 다운로드 허용
- docx QA v2의 창 길이·문서당 질문 수·근거 형식
- srt 번역 평가에서 CometKiwi 사용 가능 여부
