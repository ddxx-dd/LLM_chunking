# 구조 개편 구현 계획 (v3) — 공통 JSON 파이프라인

> 상태: **계획서(구현 전)**. 이 문서는 코드를 바꾸지 않는다. 작성 기준일 2026-10-02.
> 작업 규칙: 모든 작업은 `.venv311` 안에서만 한다. 서버 전역 환경은 건드리지 않는다(CLAUDE.md).
> 개정: v3.1 (2026-10-02) — 수정 1~7과 확정 결정 반영. 바뀐 절은 문서 맨 끝 "개정 이력" 참고.

## 0. 이 문서의 목적과 이미 정해진 결정

현재 코드(`src/`)는 트랙(docx/pdf/srt)마다 `loader → pipeline → run` 이 따로 있고, 로더가 곧바로
**마크다운 문자열**(Docling 출력)을 만들어 청커에 넘긴다. 이를 아래 흐름으로 바꾼다. **청킹·검색·번역의
핵심 알고리즘은 한 줄도 바꾸지 않는다.**

```
원본(raw) ─① 전처리(Docling)→ 공통 JSON ─② 청킹→ chunks.jsonl ─③ 인덱싱→ index ─④ 평가→ results
                                  │                                    ▲
                                  └──────── qa/*.jsonl (질문·정답·근거) ┘
```

### 확정된 결정 (사용자 지시 반영)

| 항목 | 결정 |
|---|---|
| docx_corpus QA | 다른 데이터셋처럼 **QA 세트를 새로 만든다**(7절). |
| 번역 범위 | **srt만**. 기존 docx/pdf 번역(`translate.py`, `llm.translate_batch`)은 `legacy/`로 보낸다. |
| docx 폴더 | `ko/en` 구분을 없애고 **주제(topic)별 폴더**(`education/`, `government/` …). |
| vectara `corpus/*.json` | **보관 + 정답 근거 텍스트 추출용으로 활용**(7.1절 추천안). |
| `heading_path` | JSON에는 **저장만** 한다. 청크에 제목 경로를 붙이는 기능은 **나중에 구현**(14절). |
| 청커 | fixed = LangChain 기본, semantic = LangChain `SemanticChunker` + 우리 변형(원문 오프셋 복원·토큰 후처리, 이미 구현됨), smart = 구조 + 의미 전략(pdf/docx) / 큐 묶음 + 유사도 하락 지점 분할(srt). |
| fixed 크기 | 다른 청커와 맞추기 위해 **목표 토큰 수 → 환산 글자 수**로 정한다. 환산 비율은 전처리된 문서 표본에서 데이터셋(언어)별로 **측정**한다(5.4.1절). 튜닝 후보도 목표 토큰 300·400·512. |
| 평가 지표 | 데이터셋마다 **대표 지표 하나**(docx=`evidence_recall@5`, vectara=`section_hit@5`, allganize=`page_hit@5`) + MRR. 서로 다른 지표를 평균 내지 않는다(6.2절). 결과 표에 청커별 평균 청크 토큰 수와 top-5 총 토큰 수를 함께 적는다. |
| 튜닝 후보 수 | 세 청커 모두 **9개**로 같게 한다(8절). srt는 청커마다 `max_tokens` 3개. |
| 작업 순서 | **데이터 이동과 `src/ → legacy/` 이동을 1단계**에서 한다. 새 `src/`는 빈 상태에서 작성(12절). |
| 인덱스 캐시 | 해시 검사 없이 "폴더가 있으면 불러오고 `--force`면 다시 만든다". 전처리를 다시 하면 하위 산출물을 정리한다(3.7절). |
| allganize 이미지 질문 | **eval에서 제외**(`excluded: true`). 평가 대상 211개 → 166개(7.2절). |
| `tasks/qa.py` 프롬프트 | 데이터셋 언어(`lang`)를 따른다(9.1절). |
| docx_corpus QA | 최종 **300개 이상**, 수동 검수 **10%**(7.3절). |
| srt tune | 영화 **1편**(8절). |

### 현재 데이터 (2026-10-02 기준)

| 데이터셋 | 형식 | 언어 | 문서 | QA |
|---|---|---|---|---|
| vectara | pdf | en | 100편(arXiv) | 559개 (`text` 529 / `text-table` 30, 이미지 질문 제외) |
| allganize | pdf | ko | 45개 | 211개 (원본 HF 300행 중 로컬 PDF에 해당하는 행). 이미지 질문 45개를 제외한 **166개**로 평가 |
| docx_corpus | docx | en | 100개(주제 8개) | **없음 → 새로 생성** |
| srt | srt | en/ko | 영화 5편 × 양방향 | 없음(참조 자막으로 채점) |

---

## 1. 설계 원칙 (깨지면 안 되는 불변 조건)

1. **평문 + 오프셋 불변.** 청커는 지금처럼 *평문 문자열 하나*를 받아 `text[start:end]` 리터럴 슬라이스로 자른다.
   (`chunkers.py` 상단 주석: `add_start_index`가 정확해야 srt 타임스탬프 매핑·번역 매핑이 동작한다.)
2. **JSON → 평문 변환은 결정적이다.** `to_text(doc_json)`은 같은 JSON이면 항상 같은 평문과 같은 블록 오프셋을 낸다.
3. **블록 경계 일치.** `to_text`가 기록한 블록 오프셋(`block_spans`)이 곧 SmartChunker가 쓰는 `(start, end, kind)`이다.
   평문을 정규식으로 다시 파싱하지 않는다(기존 `detect_blocks`는 새 코드에 가져오지 않는다). 올바름은 12절의 **자체 일관성 테스트**로 확인한다.
4. **동일 조건 비교.** 세 청커는 같은 JSON, 같은 토큰 카운터, 같은 임베딩 모델, 같은 인덱스/검색 코드를 쓴다. 다른 것은 청킹 함수뿐이다.
5. **재현 가능성.** 청크·인덱스는 `(입력 JSON, chunker, 파라미터, 임베딩 모델)`로 완전히 재생성 가능하다. 시드는 고정한다.
   캐시가 낡는 문제는 해시 검사가 아니라 "전처리를 다시 하면 하위 산출물을 정리한다"는 규칙으로 막는다(3.7절).

---

## 2. 폴더 구조

```
LLM_chunking/
├─ data/
│  ├─ raw/                                   원본 (git 제외)
│  │   ├─ vectara_pdf/            *.pdf (100)                  ← data/vectara_pdf/*.pdf
│  │   │   └─ _meta/              corpus/<paper_id>.json (HF 섹션 JSON, 7.1절), queries/answers/qrels.json 원본
│  │   ├─ allganize_pdf/<domain>/ *.pdf (45)                   ← data/allganize_pdf/pdf_allganize/<domain>/
│  │   │   └─ _meta/              documents.csv, rag_evaluation_result.csv, README.md 원본
│  │   ├─ docx_corpus/<topic>/    *.docx (100)                 ← data/docx_corpus/docx/<topic>/
│  │   └─ srt_eng/  srt_kor/                                   ← data/srt_eng, data/srt_kor
│  ├─ processed/<dataset>/<doc_id>.json       공통 JSON (git 제외)
│  ├─ chunks/<dataset>/<chunker>__<설정>.jsonl  청크 (git 제외)
│  ├─ index/<dataset>/<chunker>__<설정>/        벡터 인덱스 캐시 (git 제외)
│  ├─ qa/<dataset>.jsonl                      질문·정답·근거 (git 포함)
│  └─ manifests/<dataset>.json                문서 목록·메타·tune/eval 분할 (git 포함)
│
├─ src/
│  ├─ common/
│  │   ├─ config.py       DATASETS 표, 경로, 모델 이름, 청커별 기본 설정
│  │   ├─ models.py       get_embeddings(), get_token_counter(), get_llm()
│  │   └─ docjson.py      공통 JSON 읽기/쓰기, to_text(), block_spans(), to_lc_document()
│  ├─ preprocess/         ① 원본 → 공통 JSON (포맷마다 다른 유일한 곳)
│  │   ├─ pdf_prep.py   docx_prep.py   srt_prep.py
│  ├─ chunking/           ② 공통 JSON → 청크 (실험 변수)
│  │   ├─ splitters.py    chunk(doc, chunker, **params) -> list[Chunk]  (fixed/semantic 포함)
│  │   ├─ smart.py        SmartChunker 본체 (chunkers.py에서 분리)
│  │   └─ ratio.py        글자/토큰 비율 측정 (fixed 글자 수 환산, 5.4.1절)
│  ├─ indexing/           ③ 청크 → 벡터 인덱스
│  │   └─ index.py        build_index(), load_index(), get_retriever()
│  ├─ qa_build/           QA 생성 (docx_corpus 전용, 7.3절)         ← 추가
│  │   └─ gen_docx_qa.py
│  └─ tasks/              ④ 평가
│      ├─ retrieval.py    검색 지표 (hit@k, MRR, F1 + 근거/페이지 지표)
│      ├─ qa.py           검색 + Gemma 답변 + 정답 판정 (구 summary.py)
│      └─ translate_srt.py  청크 단위 번역 + 큐 단위 채점
│
├─ scripts/               실행 진입점 (로직 없음, src를 호출만)
│  ├─ preprocess.py   --dataset vectara
│  ├─ measure_ratio.py --dataset vectara                       (글자/토큰 비율 측정, 5.4.1절)
│  ├─ chunk_stats.py  --dataset vectara --chunker smart --max 400
│  ├─ run.py          --dataset vectara --task retrieval --chunker smart --max 400
│  ├─ tune.py         --dataset vectara --chunker smart        (tune 세트 20%)
│  ├─ build_qa.py     --dataset docx_corpus                    (7.3절)
│  └─ download_models.py  bge-m3, Gemma 캐시 받기              (11.2절)
│
├─ tests/                 자체 일관성 테스트 (기존 코드 출력과 비교하지 않음)   ← 추가
│  └─ fixtures/           5.3절 예시 A·B 같은 작은 고정 입력
├─ results/<dataset>/<task>/<chunker>__<설정>.json     (낡은 결과는 results/_stale/로 이동, 3.7절)
├─ docs/                  설계 메모, 진행 기록 (이 문서 포함)
├─ legacy/                기존 v2 코드 (참고용, import 금지)
└─ CLAUDE.md  README.md  requirements.txt  env.sh        (env.sh: HF_HOME 등, 11.1절)
```

### 2.1 파일 이동표 (현재 → 새 위치) — 모두 12절 **1단계**에서 한다

"현재" 열은 **2026-10-02 현재 폴더 이름** 기준이다. `data/vectara_ragbench/`와 `data/allganize/`는 이미 `data/vectara_pdf/`, `data/allganize_pdf/`로 이름이 바뀌어 있다.
원본 JSON/CSV는 1단계에서 `_meta/`로 **원형 그대로** 옮기고, `qa/*.jsonl` 변환은 7단계(`build_qa.py`)에서 한다.

| 현재 | 새 위치 | 비고 |
|---|---|---|
| `data/vectara_pdf/*.pdf` | `data/raw/vectara_pdf/` | 100개 |
| `data/vectara_pdf/corpus/*.json` | `data/raw/vectara_pdf/_meta/corpus/` | 7.1절 |
| `data/vectara_pdf/{queries,answers,qrels}.json` | `data/raw/vectara_pdf/_meta/` | 7단계에서 `data/qa/vectara.jsonl`로 변환 |
| `data/vectara_pdf/manifest_100.json` | `data/manifests/vectara.json` | 필드 유지, `split`은 7단계에서 추가 |
| `data/allganize_pdf/pdf_allganize/<domain>/*.pdf` | `data/raw/allganize_pdf/<domain>/` | 45개 |
| `data/allganize_pdf/{documents,rag_evaluation_result}.csv`, `README.md` | `data/raw/allganize_pdf/_meta/` | `data/manifests/allganize.json`(문서 목록)은 2단계에서 `documents.csv`로 생성 완료, `data/qa/allganize.jsonl`은 7단계. 파일명 매핑(`file_name`↔`orig_file_name`) 보존 |
| `data/docx_corpus/docx/<topic>/*.docx` | `data/raw/docx_corpus/<topic>/` | 100개 |
| `data/docx_corpus/docx/selected.json` | `data/manifests/docx_corpus.json` | **`path` 필드를 새 경로로 다시 쓴다** |
| `data/srt_eng/*.srt`, `data/srt_kor/*.srt` | `data/raw/srt_eng/`, `data/raw/srt_kor/` | 영화 쌍은 `manifests/srt.json`(2단계에서 생성)에 |
| `src/*` 전체 | `legacy/` | 새 `src/`는 빈 상태에서 작성 |
| `results/*` (기존 실행 결과) | `legacy/results_v2/` | 삭제하지 않음 |

이동 **전에** `data/` 파일 목록과 파일 개수(크기 포함)를 `docs/data_inventory_before_move.txt`에 남긴다. 이동 후 같은 방식으로 세어 개수·크기가 같은지 대조한다.

### 2.2 git 정책

- 추적: `src/`, `scripts/`, `tests/fixtures/`(작은 파일만), `docs/`, `legacy/`(코드), `data/qa/`, `data/manifests/`, `results/` 아래 **요약 파일만**(`results/**/summary*.json`).
- 제외: `data/raw/`, `data/processed/`, `data/chunks/`, `data/index/`, `results/`의 나머지, `results/_stale/`, `legacy/results_v2/`.
- **현재 상태 확인(2026-10-02)**: HEAD는 `backup from new server`(3a09235)이고, 로컬 설정 `.git/info/exclude`에 `data/`가 들어 있어 `data/` 전체가 추적 대상이 아니다. 추적되는 것은 옛 `.gitkeep` 4개뿐이다(옛 폴더 이름 기준이라 현재는 삭제된 것으로 표시됨).
- **추적 방식 확정(2026-10-02, 임시 저장소 실험)**: `.git/info/exclude`의 `data/`는 `.gitignore`의 `!/data/`, `/data/*`, `!/data/qa/`, `!/data/manifests/` 네 줄로 이길 수 있다. 이 규칙으로 `data/qa`·`data/manifests`는 추적 가능하고 `data/raw`·`data/processed`·`data/chunks`·`data/index`는 무시된다(실제 저장소에서도 `git check-ignore -v`로 확인). `.git/info/exclude`는 수정하지 않았다.
- `.gitignore` 정리(1단계): 옛 폴더 이름 규칙(`data/allganize/*`, `data/vectara_ragbench/*`, `.gitkeep`)을 지우고, `results/**`는 무시하되 `!results/**/summary*.json`만 예외로 둔다. `.python/`, `.tools/`, `.venv311/`는 이미 들어 있다.

---

## 3. 데이터 형식 (스키마)

### 3.1 공통 JSON — `data/processed/<dataset>/<doc_id>.json`

```json
{
  "doc_id": "2405.13879v3",
  "dataset": "vectara",
  "fmt": "pdf",
  "lang": "en",
  "title": "FACT or Fiction: Can Truthful Mechanisms Eliminate Federated Free Riding?",
  "source_file": "data/raw/vectara_pdf/2405.13879v3.pdf",
  "parser": "docling-standard",
  "parser_options": {"do_ocr": false, "table_mode": "FAST"},
  "n_pages": 20,
  "blocks": [
    {"id": "b0", "type": "heading",   "level": 1, "text": "1 Introduction",
     "heading_path": ["1 Introduction"], "page": 1},
    {"id": "b1", "type": "paragraph", "text": "Federated learning lets ...",
     "heading_path": ["1 Introduction"], "page": 1},
    {"id": "b2", "type": "table", "text": "| Client | Effort | Reward |\n| - | - | - |\n| A | 0.9 | 0.45 |",
     "heading_path": ["2 Method"], "page": 3},
    {"id": "b3", "type": "paragraph", "text": "Table 1: Rewards by effort.", "role": "caption", "for": "b2",
     "heading_path": ["2 Method"], "page": 3},
    {"id": "b4", "type": "list_item", "text": "first item", "heading_path": ["2 Method"], "page": 3},
    {"id": "b5", "type": "footnote", "text": "* Corresponding author: ...", "after_block_id": "b1",
     "heading_path": ["1 Introduction"], "page": 1}
  ]
}
```

srt는 블록이 큐다.

```json
{"doc_id": "Deadpool.en", "dataset": "srt", "fmt": "srt", "lang": "en", "title": "Deadpool", "parser": "regex-srt",
 "blocks": [
   {"id": "c1", "type": "cue", "text": "Kinda lonesome back here.", "index": 1, "start": 180.430, "end": 182.850},
   {"id": "c2", "type": "cue", "text": "Yeah, little help.",        "index": 2, "start": 183.308, "end": 184.348}
 ]}
```

- `type` 집합: `heading | paragraph | table | list_item | formula | code | footnote | cue`. 표 캡션은 `table`의 `caption` 필드가 아니라 **별도 `paragraph` 블록**(`role:"caption"`, `for:<표 id>`)이다 — Docling이 `CAPTION`을 별도 항목으로 내놓고 원문 순서(표 앞/뒤)를 따르기 때문이다. `footnote`(각주)는 전처리가 **블록 목록 맨 끝에 모아** 두고 원래 위치를 `after_block_id`로 남긴다(A안).
  예시의 4종에 `list_item/formula/code`를 더한 이유는 현재 마크다운에 목록·수식(`<!-- formula-not-decoded -->`)이 그대로 섞여 나오기 때문이다.
- `heading_path`: 제목 스택(상위→현재). **저장은 하되 청킹에는 쓰지 않는다**(14절).
- `page`: pdf만 값이 있다(Docling `prov`). docx·srt는 `null`.
- `id`는 문서 내 유일, 순서대로 부여. `doc_id` 규칙: vectara=arXiv id, allganize=`documents.csv`의 `file_name` 줄기, docx_corpus=파일명 줄기(끝 8자리 해시 포함), srt=`<영화키>.<en|ko>`.

### 3.2 청크 — `data/chunks/<dataset>/<chunker>__<설정>.jsonl` (한 줄 = 청크 하나)

```json
{"chunk_id": "2405.13879v3#0007", "doc_id": "2405.13879v3", "dataset": "vectara",
 "chunker": "smart", "params": {"min": 100, "max": 400, "window": 2},
 "start": 5120, "end": 7388, "text": "## 2 Method\n\nWe propose ...",
 "n_chars": 2268, "n_tokens": 371,
 "block_ids": ["b4", "b5", "b6"], "partial_start": false, "partial_end": false,
 "page_start": 3, "page_end": 3,
 "t_start": null, "t_end": null,
 "heading_path": ["2 Method"]}
```

- `start/end`: `to_text(doc)` 평문 기준 오프셋(= 현재 `metadata["start_index"]`).
- `partial_start/partial_end`: 청크가 블록 **중간에서** 시작/끝나면 true. 블록 단위 지표라서 *문장 경계에서 자른 경우*와 *문장 중간에서 자른 경우*를 구분하지 못한다 → 문장 절단은 5.5절 `sentence_cut_rate`가 따로 잰다. overlap이 있는 fixed에서는 인접 청크가 겹친다.
- `t_start/t_end`: srt 전용, 청크에 걸친 첫 큐의 시작과 마지막 큐의 끝(초).
- `heading_path`: 청크 첫 블록의 값. 저장만 한다.
- 파일 이름의 `<설정>`은 사람이 읽는 파라미터 요약(예: `fixed__t400`, `fixed__t400-o10`(overlap 10%), `fixed__t400-nostrip`, `semantic__p90-min100-max400`, `smart__min100-max400-w2`). fixed의 `t400`은 **목표 토큰 수**이고, 환산한 글자 수와 overlap은 각 줄의 `params`에 `{"target_tokens": 400, "chunk_size_chars": 1020, "overlap_ratio": 0.0}`처럼 기록한다. 정확한 값은 `params`가 기준이다.

### 3.3 QA — `data/qa/<dataset>.jsonl`

```json
{"qid": "283afa84-f0c8-40a7-a6f1-fb2a6b97c761", "dataset": "vectara",
 "question": "How does uncertainty in data affect standard quadratic optimization problems?",
 "answer": "Uncertainty in data affects StQPs by introducing variability ...",
 "doc_id": "2411.14884v3", "qtype": "abstractive", "source": "text",
 "evidence": {"kind": "section_text", "section_id": 1, "text": "..."},
 "gold_page": null, "excluded": false, "split": "eval"}
```

- `qtype`: `extractive | abstractive | table`(vectara의 `type`/`source`, allganize의 `context_type`, docx 생성 질문의 유형을 모두 이 칸에 맞춘다).
- `evidence.kind`: `section_text`(vectara) · `span`(docx_corpus 생성: `{block_ids, start, end}`) · `none`(allganize, 대신 `gold_page`).
- `gold_page`: allganize의 `target_page_no`.
- `excluded`: `true`면 평가·튜닝에서 모두 뺀다(allganize 이미지 질문). 이 질문은 `split`을 `null`로 둔다. 해당 문서는 다른 질문 때문에 코퍼스에는 그대로 들어간다.
- `split`: `tune | eval`(8절).

### 3.4 manifest — `data/manifests/<dataset>.json`

문서별 메타(기존 `manifest_100.json`·`selected.json` 필드 유지) + `split` + 생성 정보.
srt는 `{"movie": "Deadpool", "en": "Deadpool.en", "ko": "Deadpool.ko"}` 쌍 목록(현재 `config.SUBTITLE_DATASETS`가 하던 일).

### 3.5 인덱스 캐시 — `data/index/<dataset>/<chunker>__<설정>/`

- 저장 방식(2026-10-06 개정): **FAISS `IndexFlatIP`(정확 검색)** - `index.faiss`, `chunk_ids.json`, `info.json`. `faiss-cpu`는 `.venv311`에만 설치한다. 원래 계획은 LangChain `InMemoryVectorStore.dump/load`였으나 dump가 벡터를 들여쓴 JSON으로 저장해 파일이 매우 커서 쓰지 않았다(그 사이 numpy 직접 검색을 거쳐 FAISS로 옮김). 청커 비교 실험이라 검색 근사 오차가 섞이지 않도록 근사 인덱스(HNSW, IVF)는 쓰지 않는다.
- 규칙은 단순하다: **폴더가 있으면 불러오고, `--force`면 다시 만든다.** 해시 검사와 자동 재생성은 하지 않는다.
- `info.json`(`embed_model`, `n_chunks`, `created_at`)은 사람이 확인하는 용도이고 검증에는 쓰지 않는다.
- 청크 파일이 바뀐 뒤에는 같은 폴더 이름의 옛 인덱스가 남을 수 있으므로 3.7절 규칙으로 정리한다.

### 3.6 결과 — `results/<dataset>/<task>/<chunker>__<설정>.json`

이어하기 규칙 유지: 같은 파일이 있으면 건너뛴다(`--force`로 덮어쓰기). 이 때문에 전처리를 다시 한 뒤에는 옛 결과가 새 실행을 막으므로 3.7절 규칙이 필요하다.

### 3.7 캐시 일관성 규칙 — 전처리를 다시 했을 때

각 단계는 "파일이 있으면 다시 만들지 않는다". 파일 이름은 청커·파라미터로 정해지고 입력(전처리 결과)이 바뀌었는지는 이름에 없다.
그래서 전처리 결과를 바꾸면 그 뒤 단계 파일이 **조용히 낡는다**.

```
processed JSON ─▶ chunks/ ─▶ index/ ─▶ results/        (앞이 바뀌면 뒤는 전부 낡음)
      └──────────▶ qa/docx_corpus.jsonl의 근거 스팬(오프셋)
```

`scripts/preprocess.py --dataset X --force`는 끝난 뒤 다음을 처리한다(`--limit`로 일부 문서만 다시 만든 경우도 동일 — 청크·인덱스는 코퍼스 단위라 일부만 새것이면 일관성이 깨진다).

| 대상 | 처리 | 이유 |
|---|---|---|
| `data/chunks/X/` | 삭제 | 옛 JSON 기준 오프셋 |
| `data/index/X/` | 삭제 | 옛 청크로 만든 인덱스 |
| `results/X/` | `results/_stale/X_<날짜>/`로 이동(삭제 아님) | 이어하기가 "이미 결과 있음"으로 새 실행을 건너뛰는 것을 막음 |
| `data/qa/docx_corpus.jsonl`의 근거 스팬 | 삭제하지 않고 **경고 출력** | 근거를 `to_text` 오프셋으로 저장하므로 전처리가 바뀌면 위치가 틀어짐 → `build_qa.py --dataset docx_corpus`로 근거를 다시 계산 |

- 권장(선택): 청크 파일을 불러올 때 모든 청크가 `text == to_text(doc)[start:end]`인지 확인하는 검사를 둔다. JSON을 직접 편집한 경우처럼 `--force`를 거치지 않은 변경을 청크 단계에서 잡아낸다(해시는 쓰지 않는다).

---

## 4. 전처리 상세 (① raw → 공통 JSON)

### 4.1 공통: 평문 변환과 블록 오프셋 (`common/docjson.py`)

```python
def render(doc) -> tuple[str, list[Piece]]:
    """블록을 평문으로 이어 붙이고, 조각(Piece)별 (start, end, kind, block_ids)를 함께 돌려준다."""

def to_text(doc) -> str:                 return render(doc)[0]
def block_spans(doc) -> list[tuple]:     # SmartChunker.sections()가 받는 (start, end, kind) 목록
    return [(p.start, p.end, p.kind) for p in render(doc)[1]]
def to_lc_document(doc) -> Document:     # srt용: metadata["units"] 포함 (mapper/번역 코드 무수정)
```

렌더 규칙(현재 로더 출력과 같게 맞추는 것이 목표):

| 블록 | 평문 | Piece.kind |
|---|---|---|
| heading(level n) | `"#"*n + " " + text` | `heading` |
| paragraph / formula / code / footnote | `text` | `para` |
| list_item | 항목마다 별도 조각(빈 줄로 구분). 참고문헌 수백 개가 쪼개지지 않는 한 조각이 되는 것을 막고 smart가 항목 사이에서 자를 수 있게 한다 | `para` |
| table | 표 마크다운. (`caption` 필드가 있으면 그 한 줄을 별도 조각으로 앞에 두는 규칙이 있으나 전처리는 이 필드를 쓰지 않는다) | `table` |
| cue (srt) | `text` | `cue` |

- 마크다운 계열(pdf/docx): 조각 사이는 `"\n\n"`. fixed/semantic이 평문을 문단 단위로 다룰 때와 사람이 읽을 때 같은 모양이 되도록 한다.
- srt: 조각 사이는 `SEP = "\n"`(현재 `srt/loader.py`와 동일). 각 큐 뒤에 `SEP`가 붙고 마지막에도 붙는다.
- 불변식(자체 검증): ① `to_text(doc)[p.start:p.end]`가 각 조각의 텍스트와 같다 ② 조각은 순서대로이며 서로 겹치지 않는다 ③ `block_ids`가 JSON의 블록과 1:1로 대응한다.
  ④ 같은 JSON에서 `to_text`를 두 번 불러도 결과가 같다. 기존 로더의 마크다운과 글자 단위로 같을 필요는 없다(새 코드가 기준이다).

### 4.2 `preprocess/pdf_prep.py`

현재 `pdf_track/loader.py`를 줄 단위로 대응시킨다(구현 첫 작업: `build_markdown` 전체를 읽고 표로 정리).

| 현재 동작 | 새 코드에서 |
|---|---|
| `torch.backends.cudnn.enabled = False` (서버 cuDNN 불일치 우회) | 그대로 유지 |
| `PdfPipelineOptions(do_ocr=False)`, `TableFormerMode.FAST` | 그대로 유지, `parser_options`에 기록 |
| `_drop_corner_items`: 쪽 위/아래 5% 안의 머리말·쪽번호 제거(그림·표 제외) | 그대로 호출 |
| `_PdfHeadingLevel`: 제목 번호(`3`, `3.1`)로 수준 복원, `Abstract/Introduction/...`는 1단계 | 그대로 → `heading.level` |
| `_SKIP_LABELS`: PICTURE/PAGE_HEADER/PAGE_FOOTER 제외 | 그대로 |
| Docling "inline 그룹" 항목을 한 줄로 합침(`_inline_group_ref`) | 같은 그룹의 항목은 **한 paragraph 블록**으로 합침 |
| 3칸 이상 공백 압축(`_WS_COLLAPSE_RE`) | 블록 `text`에 적용 |
| 표 → 마크다운 표 | `table` 블록의 `text`. **캡션 없이, 패딩 없는 compact 형식**(아래) |

- 추가로 얻는 것: 각 블록의 `page`(Docling `prov[0].page_no`, 1부터), 제목 스택(`heading_path`).
- **2단계 구현에서 확정한 규칙** (`src/preprocess/_docling.py`, `pdf_prep.py`):
  - `FOOTNOTE`는 따로 모아 **맨 끝**에 붙이고 `after_block_id`에 원래 위치를 남긴다(A안). 본문 문장 사이에 끼는 노이즈가 사라진다. 각주는 마지막 섹션(보통 `References`)의 끝에 붙게 된다.
  - 표 캡션: Docling이 `CAPTION`을 표 바로 앞/뒤의 **별도 항목**으로 내놓고, 표 마크다운 내보내기도 기본으로 캡션을 포함해서 그대로 쓰면 캡션이 **두 번** 나온다(실측). 그래서 표 직렬화에서 캡션을 빼고(`MarkdownParams(labels - {CAPTION})`), `CAPTION`은 문단 블록으로 원문 순서대로 둔다.
  - 표는 `compact_tables=True`: 열 폭 패딩 공백·구분선이 사라져 표 크기가 약 절반이 된다(실측 679자 → 348자). 내용은 같다.
  - `LIST_ITEM`은 항목마다 별도 블록(기존 pdf 로더와 같은 방향). 3칸 이상 공백 압축은 모든 블록에 적용한다.
  - `PICTURE`·`CHART`·`PAGE_HEADER`·`PAGE_FOOTER`는 제외(실측: 이번 문서들에서 `PAGE_HEADER/FOOTER`는 0개라 쪽 상/하단 5% 규칙이 실제 거름망이다).
  - **PDF 정제(구현·적용 완료, 2026-10-03)**: Docling 기본 PDF 텍스트 엔진(`docling-parse`)이 일부 PDF에서 본문 한 줄을 통째로 읽지 못하는 문제가 있다(allganize `journal-31-3-308`의 초록 한 줄 누락으로 확인; 레이아웃 모델·우리 코드가 아니라 텍스트 추출 단계의 문제이고 파서 설정으로 끌 수 없다 — [docling-parse #371](https://github.com/docling-project/docling-parse/issues/371) 참고). PyMuPDF로 PDF 내용 스트림을 정규화해 다시 저장하면(`clean=True, garbage=4, deflate=True`) 같은 줄이 읽힌다(실측 글자 +47, 표본 24편에서 표·제목 개수 변화 0·1% 이상 감소 0편). 그래서 `convert_pdf`는 기본으로 정제본을 임시로 만들어(`data/processed/_tmp_clean/`, 변환 직후 삭제) 변환한다.
    - 정제가 실패하면(암호화·손상·쪽 수 불일치) 원본을 그대로 변환하고 `log.pdf_clean.error`에 이유를 남긴다. 정제 여부와 옵션은 문서 JSON의 `log.pdf_clean`, `parser_options.pdf_clean`에 기록된다.
    - `scripts/preprocess.py --no-pdf-clean`은 원본 그대로 변환한다(A/B 비교용). 이미 있던 JSON을 `--force`로 다시 만들면 이전 결과 대비 글자 수 차이(`delta_chars`)를 `_report.json`에 남기고 마지막에 증가·감소 문서 수를 출력한다 — 정제의 실제 효과를 전체 PDF에서 확인하는 용도다.
    - 의존성: `pymupdf`(AGPL-3.0)를 `requirements.txt`에 명시했다(기존에는 `pdf2docx`의 간접 의존성으로만 설치돼 있었다).
    - 결과: vectara 100편·allganize 45편을 정제본으로 다시 변환했다(정제 실패 0건). 누락된 줄이 복구됐고 겹쳐 그려진 PDF의 **중복 블록도 크게 줄었다**(4.6절).
- 문서 하나당 변환 시간이 크므로 **이미 JSON이 있으면 건너뛴다**(`--force`로 재생성). 100편 변환은 한 번만 하면 된다.

### 4.3 `preprocess/docx_prep.py`

- 현재: 기본 `DocumentConverter()` → `export_to_markdown(image_placeholder="")` → 글자 없는 제목 줄 제거(`^#+[ \t]*\n+`).
- 새 코드: 같은 변환기로 항목을 순회해 블록 생성. 빈 제목은 만들지 않고, 그림은 건너뛴다.
- `page`는 `null`(docx에는 쪽 정보가 없다). 제목 수준은 Word 스타일(Heading 1–5)을 Docling이 준 값 그대로 쓰되, **문서에 `TITLE`이 있으면 `TITLE`을 1단계로 두고 `SECTION_HEADER`를 한 단계 내린다**(기존 Docling 마크다운 내보내기와 같은 위계, 최대 6). 각주는 이번 표본(docx 100개 전부)에서 0개였다.
- 입력은 `data/raw/docx_corpus/<topic>/*.docx`. `topic`은 manifest에도 남긴다.

### 4.4 `preprocess/srt_prep.py`

- 현재 `srt/loader.py`의 `read_text`(인코딩 순차 시도), `CUE_RE`, `TAG_RE`, `normalize`(NFC·공백·특수 공백 정리)를 그대로 옮긴다. 태그 제거 후 빈 큐는 버리고 개수를 `log`에 기록하는 동작도 유지한다.
- 큐 → `cue` 블록: `index`(원래 번호), `start/end`(초, `_to_sec`).
- `to_lc_document(doc)`는 현재 `_Builder.done()`이 만들던 `Document(page_content, metadata{name, fmt, log, units[{start, end, kind, meta{index, t_start, t_end}}]})`를 그대로 재현한다. 그래서 `mapper.py`, `timestamp_align.py`, `subtitle_translate.py`는 **수정 없이** 재사용된다.

### 4.5 `scripts/preprocess.py`

```
python scripts/preprocess.py --dataset vectara [--limit N] [--force]
```
`manifests/<dataset>.json`의 문서 목록 순서로 변환 → `data/processed/<dataset>/<doc_id>.json`. 변환 실패 문서는 로그에 남기고 계속 진행한다.
`--force`는 JSON을 다시 만들고 **하위 산출물을 3.7절 규칙대로 정리**한다. `--force` 없이는 이미 있는 JSON을 건너뛴다.

### 4.6 2단계 실행 결과 (2026-10-02)

> vectara·allganize는 **2026-10-03에 PDF 정제(4.2절)를 적용해 다시 변환한 결과**다(이전 JSON은 덮어씀). docx·srt는 정제와 무관해 그대로다. 아래 표의 vectara·allganize 행과 "PDF 정제 적용 결과" 항목이 재변환 이후 값이다.

`scripts/preprocess.py`로 전체 변환, `scripts/measure_ratio.py`로 비율 측정, `scripts/annotate_manifests.py`로 매니페스트에 실측 필드 추가. 변환 서버: RTX A5000, 모든 문서 **오류 0건·실패 0건**.

| 데이터셋 | 문서 | ok / warn | 블록 | 변환 시간 합 | 비고 |
|---|---|---|---|---|---|
| vectara | 100 | 70 / 30 | 27,811 (제목 2,274 · 문단 18,489 · 목록 6,093 · 표 332 · 각주 580 · 코드 43) | 866초 (정제 166초 포함) | 각주가 있는 논문 77편 |
| allganize | 45 | 32 / 13 | 10,423 (제목 2,089 · 문단 3,719 · 목록 4,271 · 표 195 · 각주 144 · 코드 5) | 341초 (정제 17초 포함) | 각주가 있는 문서 23개 |
| docx_corpus | 100 | 72 / 28 | 34,333 (제목 4,204 · 문단 23,413 · 목록 6,658 · 표 50 · 수식 8) | 479초 | 각주 0개 |
| srt | 10 (영화 5 × 2언어) | 10 / 0 | 10,592 큐 | 1초 미만 | 큐 개수가 원본 `-->` 개수와 일치 |

- **경고의 내용** (`data/processed/<dataset>/_report.json`): vectara 30건 중 25건은 "제목 5개 이상인데 수준이 전부 같다"(제목에 번호가 없는 논문은 기존 규칙상 전부 1단계 — 기존 코드와 같은 한계), 4건은 수준 건너뜀, 1건은 글이 거의 없는 쪽이 많음. allganize 13건은 제목 수준 단일(10)·글 없는 쪽 많음(6; 그림 위주 문서 의심, OCR을 끄고 있음)·글자 수 적음(1). docx 28건은 제목 수준 건너뜀(24)·첫 제목이 2단계(14)로, **원본 Word 스타일이 그렇다**(예: 법령 문서는 `Heading 2` 바로 다음에 `Heading 5`).
- **docx 내용 보존**: 원본 Word 글자 수 대비 JSON 글자 수가 최소 0.946 ~ 최대 1.115(중앙 1.001), 0.9 미만 0개. 제목 수는 중앙 1.00배. Word 표 개수와 JSON 표 개수가 다른 문서가 8개(JSON이 더 적음 — Docling이 일부 표를 표로 읽지 못함, 글자는 보존됨).
- **vectara `manifest`의 옛 `n_tables`는 정확하지 않았다.** 첫 50편의 옛 값 합계 20 vs 이번 변환 157(예: 2404.09358v3은 0 → 15). 새 50편은 대체로 일치(191 → 175). HF 원본 JSON의 표 수와는 72/100편이 일치한다. 옛 값을 덮어쓰지 않고 `n_tables_docling`을 새 필드로 추가했다. 이번 변환 기준 표가 3개 이하인 논문은 68/100편이다(10개 이상 13편).
- **가장 긴 조각**: vectara 5,971자, docx 6,453자, allganize 2,709자 — 8,000자를 넘는 조각은 없다. 목록 항목 분리와 표 compact 형식의 효과다.
- **PDF 정제 적용 결과 (2026-10-03 재변환, `delta_chars`로 이전 결과와 비교)**
  - 정제 적용 vectara 100/100, allganize 45/45, 정제 실패(원본으로 대체) 0건. 변환 오류·실패 0건, ok/warn 문서 수 변화 없음.
  - 글자 수 변화: vectara 증가 10편(+630자)·감소 6편(-2,517자)·같음 84편, allganize 증가 4편(+56자)·감소 3편(-100자)·같음 38편.
  - **누락 줄 복구**: `journal-31-3-308`의 초록 누락 줄이 복구됐다(+48자). 이 문서의 `heading_path`가 저자명으로 오염되던 문제도 함께 사라졌다.
  - **감소는 내용 손실이 아니라 중복 제거**였다(감소폭이 큰 두 문서를 정제 없이 다시 변환해 직접 비교): `2409.06325v3`은 블록 727개(같은 블록이 중복된 횟수 240) → 410개(중복 15), `2404.18869v2`는 741개(중복 205) → 463개(중복 11). 같은 줄을 겹쳐 그린 PDF에서 `docling-parse`가 줄을 복사본마다 따로 읽는 문제([#366](https://github.com/docling-project/docling-parse/issues/366))와 일치한다. allganize `KIFVIP2013-10`(-77자)은 표 텍스트의 띄어쓰기·구분 표기 차이로 내용 손실은 없었다. 나머지 소폭 감소 문서(-37자 이하, 6편)는 개별 확인하지 않았다.
  - 재변환 후 문단·목록 블록의 중복 비율(30자 이상 동일 블록)은 PDF 전 문서에서 5% 이하(최대 3.7%, 반복되는 제목·캡션 수준)다. 참고로 docx는 8개 문서가 5%를 넘는데(최대 17.6%) PDF 파서를 거치지 않으므로 정제와 무관하며 원본 Word 문서의 반복 내용으로 보이나 확인하지 않았다.
- **글/토큰 비율**: 5.4.1절 표.
- **발견한 버그와 수정**: 문서 ID가 `_`로 시작하는 allganize 문서 3개가 `_report.json` 제외 규칙에 같이 걸려 비율 측정에서 빠졌다 → 제외를 `_report.json` 한 파일로 한정하고, 측정 문서 수가 매니페스트와 다르면 실패하도록 검사를 추가했다.

---

## 5. 청킹 상세 (② 공통 JSON → 청크)

> **개정 (2026-10-06, v3.4) — semantic과 smart는 같은 엔진이고, 차이는 구조 제약 규칙뿐이다.**
> 아래 5.1·5.3·5.4의 semantic/smart 설명(LangChain `SemanticChunker` + `recover_spans`, 섹션 우선 `SmartChunker`)은 v3.3까지의 설계이고,
> 코드에서는 `chunking/legacy_smart.py`로 옮겨 보관한다(srt 설정이 아직 `SmartChunker`를 쓴다). 5.3의 예시는 그 옛 동작을 보여 주는 기록이다.
>
> **현재 구현 (pdf·docx)**
>
> | 모듈 | 역할 |
> |---|---|
> | `chunking/semantic_core.py` | 공유 엔진. 단위(문장) → (이웃 결합) 임베딩 → 코사인 거리 → 문서 percentile 임계값 → 컷(+forced) → max 초과는 최대 거리에서 쪼개기 → min 미만은 합치기. **이웃 결합은 기본 끔(`buffer_size=0`)** |
> | `chunking/units.py` | 구조 제약 규칙(`Rules`)과 `build_units`: 규칙에 따라 엔진에 넘길 단위와 forced 경계를 만든다. `caption_glue`도 여기 있다 |
> | `chunking/splitters.py` | `fixed`(LangChain 글자 수 분할), `semantic`(엔진 + 규칙 전부 off), `smart`(엔진 + 규칙 전부 on) 등록부 |
>
> | 규칙 | 켜면 | 끄면 |
> |---|---|---|
> | `heading_forced` | 제목 앞은 반드시 자른다(forced) | forced 없음 |
> | `heading_glue` | 제목을 뒤따르는 첫 문장 단위에 붙여 제목 뒤를 못 자른다(연속 제목은 함께) | 제목이 평문의 한 문장 |
> | `table_atomic` | 표 전체가 단위 하나(max를 넘어도 한 덩어리) | 표를 행마다 쪼갠다 |
> | `caption_glue` | 표와 캡션 사이를 못 자른다 | 따로 둔다 |
>
> - 제약은 엔진 안의 분기가 아니라 **엔진에 넘기는 단위 목록**으로 표현한다. 규칙을 모두 끄면 단위가 평문 전체의 문장 분할과 같아져 semantic과 청크가 동일하다(`tests/test_constraints.py`). 일부 규칙만 켜는 어블레이션은 설정(`rules`)만 바꾸면 되고 파일 이름에 `-r<H|G|T|C>`가 붙는다.
> - **semantic은 LangChain `SemanticChunker`를 호출하지 않는다.** 그 계산(이웃 결합 임베딩, 코사인 거리, `np.percentile`, `>` 판정)을 엔진에 다시 구현했고, 크기 정리 전의 경계가 LangChain 결과와 같음을 buffer 0·1·2에서 동치 테스트로 확인한다(`tests/test_semantic_engine.py`; LangChain의 기본은 buffer 1이므로 `buffer_size=1`로 설정하면 기본 동작을 재현한다). 청크는 원문 구간을 그대로 자르므로 `recover_spans`와 `semantic_dropped_chunks`는 사라졌다.
> - **정책 결정(사용자, 2026-10-06)**: 표는 쪼개지 않는다(앞뒤 문단과 한 청크에 들어갈 수 있고, max를 넘으면 단독). 제목 앞 강제 컷도 청크가 min 미만이면 합친다(합친 횟수는 `merges_across_forced`로 보고).
> - 이전 semantic 결과와 청크가 다르다: 크기 정리가 단위 경계 위에서 같은 거리값을 쓰고(문장을 다시 임베딩하지 않음) 양쪽이 min 이상이 되는 곳을 우선한다.
> - **buffer 기본값 0의 이유(2026-10-06 결정)**: 이웃을 결합하면 경계 양쪽 내용이 두 임베딩에 모두 섞여서 진짜 경계의 거리 대비가 줄고(통제된 가짜 임베딩 데이터에서 0.988 → 0.199) 양옆에 반높이 어깨가 생긴다. percentile 임계값에서 어깨가 함께 넘으면 봉우리 둘레에 컷이 여러 개 생길 수 있고, 문장별 주제 밀도가 들쭉날쭉하면 최대 위치가 한 칸 어긋날 수 있다(buffer 2에서 관찰). buffer 1이 체계적으로 한 칸 뒤에 컷을 만드는 것은 아니다(이전에 그렇게 적은 것은 한 예시에서의 과장이었다). 문장 하나만 임베딩하므로 실제 임베딩에서는 거리가 더 흔들릴 수 있다(Qu 외의 문맥 없는 문장 임베딩 한계). 실제 문서에서 제목 앞 경계(알려진 주제 전환)로 buffer·겹치지 않는 창 방식을 비교하는 진단은 아직 하지 않았다. 문장 단위 임베딩이라 문단 단위 방식보다 느리지만, 결합을 끄면 입력이 짧아 비용과 메모리 부담은 줄어든다.
> - srt: `smart`는 아직 legacy(`algorithm="legacy"`), `semantic`은 같은 엔진(규칙 off)이다. srt 전략은 나중에 정리한다.

### 5.1 인터페이스 (`chunking/splitters.py`)

```python
def build_splitter(chunker: str, dataset_cfg, params, models) -> TextSplitter:
    # fixed    -> make_fixed(chunk_size, strip_whitespace, chunk_overlap)   # chunk_size = target_tokens × 측정한 chars_per_token (5.4.1절)
    # semantic -> make_semantic(embeddings, count_tokens, percentile, min_tokens, max_tokens)
    # smart    -> make_smart(embed_model, count_tokens, mode, lang, min_tokens, max_tokens, window)

def chunk(doc, chunker, **params) -> list[Chunk]:
    text, pieces = render(doc)
    spans = splitter_spans(text)               # [(start, end), ...]  ← 기존 코드가 만든 그대로
    return [attach_meta(doc, text, pieces, s, e) for s, e in spans]
```

- `make_fixed`, `make_semantic`, `recover_spans`, `_split_oversized`, `_merge_small_spans`, `SmartTextSplitter`, `SmartChunker`의 `boundary_scores / split_section / merge_small / chunk_spans`는 **본문 그대로** 옮긴다. `smart.py`로의 분리는 코드 이동뿐이다.
- 바뀌는 곳은 딱 하나: `SmartChunker.sections()`가 입력으로 받는 `blocks`를 정규식 `detect_blocks(text, mode)` 대신 `block_spans(doc)`에서 직접 받는다. `detect_blocks`는 새 코드에 옮기지 않는다. 섹션 나누기·경계 점수·분할·병합 알고리즘은 그대로다.
- 알고리즘이 그대로인지는 5.3절 예시 A·B(입력 고정, 가짜 임베딩, 설명용 파라미터)를 `tests/fixtures/`에 고정 입력으로 두고, 이번에 실행해 얻은 청크 경계를 기대값으로 삼아 확인한다.
- 토큰 카운터는 데이터셋별 설정값이다: pdf/docx = `BAAI/bge-m3` 토크나이저, srt = Gemma 토크나이저(현재와 동일).
- fixed는 `make_fixed(chunk_size, ...)`의 `chunk_size`(글자 수)를 그대로 쓰되, 이 값을 사람이 정하지 않고 `target_tokens`와 측정한 비율에서 계산해 넘긴다. `chunk_overlap`은 글자 수(`overlap_ratio × chunk_size`)이다. 알고리즘(LangChain `CharacterTextSplitter`)은 그대로다.

### 5.2 `attach_meta` — 청크에 메타데이터 붙이기

```
오프셋 이진탐색으로 청크 [s, e)와 겹치는 Piece를 찾는다
block_ids      = 겹친 Piece들의 block_ids
partial_start  = (s > 첫 겹침 Piece.start)       # 블록 중간에서 시작
partial_end    = (e < 마지막 겹침 Piece.end)      # 블록 중간에서 끝남
page_start/end = 겹친 블록의 page 최솟값/최댓값 (pdf)
t_start/t_end  = 겹친 첫 큐 start / 마지막 큐 end (srt)
heading_path   = 첫 블록의 heading_path        # 저장만
```
이 메타데이터는 검색 점수 계산에 쓰이지 않는다(쓰이는 것은 `text`뿐). 따라서 지표 값에 영향을 주지 않는다.

### 5.3 세 청커가 실제로 어떻게 자르는가 — 실행한 예시

아래는 **설명용으로 파라미터를 줄여서 실제 청커 코드를 돌린 결과**다(임베딩은 키워드 기반 가짜 모델, 토큰 카운터는 공백 단위 단어 수).
실제 실험의 값은 pdf/docx `min 100 / max 400 토큰`(bge-m3), srt `min 128 / max 512·1024 토큰`(Gemma)이다.

#### 예시 A — 논문 한 부분 (PDF/docx 형태, 평문 1,069자)

변환된 평문 (블록은 빈 줄로 구분, `⏎⏎`로 표시):

```
[0..16)     # 1 Introduction                                   heading   (3단어)
[18..241)   Federated learning lets many clients train ...     para      (35단어)
[243..388)  Prior work assumes honest participation. ...       para      (23단어)
[390..401)  ## 2 Method                                        heading   (3단어)
[403..501)  We propose a truthful mechanism that rewards ...   para      (14단어)
[503..602)  | Client | Effort | Reward |  (표 4행)             table     (29단어)
[604..937)  The server estimates each contribution ...         para      (49단어)
[939..955)  ## 3 Experiments                                   heading   (3단어)
[957..1069) We evaluate on CIFAR-10 and FEMNIST ...            para      (18단어)
```

**① fixed (chunk_size=160자, 겹침 0)** — 글자 수만 본다. 실제 출력:

| # | 구간 | 시작 부분 → 끝 부분 | 문제 |
|---|---|---|---|
| 0 | 0–160 | `# 1 Introduction ⏎⏎Federated learning … may free ride by contributing little whil` | 단어 중간(`whil`)에서 끝남 |
| 1 | 160–320 | `e still receiving the global model. …` | 앞 청크의 `whil`+`e`가 갈라짐 |
| 2 | 320–480 | `ach client as a rational agent … ## 2 Method ⏎⏎We propose … their v` | 문장·단어 중간에서 시작/끝 |
| 3 | 480–640 | `erified contribution. ⏎⏎\| Client \| … \| C \| 0.0 \| 0.00 \| ⏎⏎The server estimates each contributi` | 표와 앞뒤 문장이 한 청크에 섞임 |
| 4 | 640–800 | `on by comparing … The reward rule is` | 문장이 끝나지 않고 잘림 |
| 5 | 800–959 | `monotone in alignment … ## 3 Experiments ⏎⏎We` | 제목이 본문과 떨어져 청크 끝에 `We`만 남음 |
| 6 | 960–1069 | `evaluate on CIFAR-10 …` | 제목 `3 Experiments`와 본문이 다른 청크 |

이 문서에서 `partial_start/partial_end`는 7개 청크 중 6~7개가 true가 된다. 이것이 "단순 글자수 분할이 문맥을 끊는다"를 숫자로 보이는 지표다(5.5절 `chunk_stats`). 다만 `partial_*`는 블록 단위라 문장 경계에서 자른 경우도 true가 되므로, 문장 중간 절단은 `sentence_cut_rate`로 따로 센다.

**② semantic (LangChain `SemanticChunker` percentile 90 + 우리 후처리, min 25 / max 60단어)** — 문장 단위(정규식 `(?<=[.?!。])\s+|\n+`)로 쪼개 인접 문장 임베딩 거리가 큰 곳에서 자른다. 실제 출력:

| # | 시작 | 단어 | 내용 |
|---|---|---|---|
| 0 | 0 | 3 | `# 1 Introduction` (제목이 혼자 남음) |
| 1 | 18 | 58 | Introduction 본문 두 문단 |
| 2 | 390 | 3 | `## 2 Method` (제목이 혼자 남음) |
| 3 | 403 | 36 | `We propose … ` + **표의 앞 3행** |
| 4 | 584 | 56 | **표의 마지막 행 `\| C \| 0.0 \| 0.00 \|`** + `The server estimates …` |
| 5 | 939 | 21 | `## 3 Experiments` + 본문 |

- 의미 유사도로 소주제(서론/방법/실험) 경계는 잘 찾는다.
- 하지만 **표를 모른다**: 표의 각 행이 줄바꿈(`\n+`)으로 "문장"이 되어 표가 청크 3과 4 사이에서 갈라졌다.
- 제목이 홀로 남는 현상은 `_merge_small_spans`의 규칙("합친 크기가 max_tokens 이하일 때만")과 거리 기반 분할의 결과다.
  → 이 동작은 **바꾸지 않는다.** 대신 `chunk_stats`가 *고아 제목 비율*, *표가 갈린 비율*을 측정해 보고서에 한계로 적는다.

**③ smart (구조 → 의미 → 병합, min 25 / max 60단어, window 2)** — 실제 출력:

```
1단계 구조: 제목에서 섹션을 나눈다
  섹션0 = [# 1 Introduction][para 35][para 23]       합계 61 > 60
  섹션1 = [## 2 Method][para 14][table 29][para 49]  합계 95 > 60
  섹션2 = [## 3 Experiments][para 18]                합계 21 ≤ 60   ← 자를 필요 없음

2단계 의미: 합계가 max를 넘는 섹션만, 경계 점수(window=2 평균 임베딩 코사인; 낮을수록 자르기 좋음)가
            가장 낮은 곳에서 재귀 분할. 단, 양쪽이 모두 min 이상인 후보만 (없으면 전체 경계 중 최저점).
  섹션0: 후보 없음(제목 3단어 / 나머지 58단어 → 3 < 25) → 전체 중 최저점 = 제목 바로 뒤
  섹션1: 후보 1개 = 표 뒤(왼쪽 46 / 오른쪽 49) → 표 뒤에서 분할

3단계 병합: 섹션 안에서만 min 미만 청크를 이웃과 합침(합친 크기 ≤ max일 때)

결과
  #0   0–16     3단어  # 1 Introduction                   ← 섹션0 분할 시 제목이 혼자 남음(아래 설명)
  #1  18–388   58단어  Introduction 본문 두 문단
  #2 390–602   46단어  ## 2 Method + 문단 + 표 (표 통째로)
  #3 604–937   49단어  The server estimates … (표 뒤 문단)
  #4 939–1069  21단어  ## 3 Experiments + 본문           ← min(25) 미만이지만 섹션 경계를 넘어 합치지 않음
```

- 표는 절대 갈라지지 않고(`table`은 쪼개지지 않는 한 조각), 제목은 자기 섹션의 첫 조각이다.
- **알려진 동작 두 가지** (코드 그대로이며 바꾸지 않는다; 측정해서 한계로 서술):
  1. 섹션이 max를 *조금* 넘는데 유효한 분할 후보가 없으면(위 섹션0) 최저점 경계로 잘라 **제목만 남는 작은 청크**가 생길 수 있다. 실제 파라미터(max 400)에서는 드물 것으로 예상하지만 `chunk_stats`의 `orphan_heading` 지표로 확인한다.
  2. `merge_small`은 섹션 안에서만 합친다(섹션을 넘어 합치면 서로 다른 절이 섞여 실측 문제가 있었다). 그래서 짧은 절은 min 미만 청크로 남는다.
- 청크 #2처럼 "제목 + 본문 + 표"가 한 청크에 같이 들어가는 것이 smart의 이점이고, 검색 때 `Method` 질문이 표와 문단을 함께 가져오게 만든다.

#### 예시 B — 자막 (Deadpool 영어 자막 1–19번 큐)

평문은 큐마다 한 줄(`SEP="\n"`), 오프셋은 `(큐 시작, 큐 끝)`:

```
 1 [00:03:00,430→00:03:02,850] Kinda lonesome back here.
 2 [00:03:03,308→00:03:04,348] Yeah, little help.
 3 [00:03:05,686→00:03:10,146] Okay, just, I have to keep my hands on the wheel.
 4 Excuse me.            5 Dopinder.           6 Nice.            7 Smells good, no?
 8 She would have made me a very agreeable wife.
 9 But, Gita's heart has been stolen by my cousin Bandhu.
10 He is as dishonorable as he is attractive.
11 Dopinder, I'm starting to think there's a reason I'm in this cab today.
12 Yeah, sir, you called for it, remember?
13 When you find it, the whole world tastes like Daffodil Daydream.
14 So you gotta hold onto love... tight!
15 Or else the whole world tastes like Mama June after hot yoga.
16 Sir, what does Miss Mama June taste like?          17 It's bad.
18 Why the fancy red suit, Mr. Pool?
19 That's because it's Christmas Day, Dopinder.
```

**fixed (chunk_size=200자, `strip_whitespace=False`)** — 실제 출력:

| 청크 | 구간 | 걸친 큐 | 설명 |
|---|---|---|---|
| 0 | 0–200 | 1–8, **9\*** | 9번 큐가 중간에서 끊김 |
| 1 | 200–400 | **9\***, 10–12, **13\*** | 9번의 뒷부분, 13번 앞부분 |
| 2 | 400–600 | **13\***, 14, 15, **16\*** | |
| 3 | 600–691 | **16\***, 17–19 | |

(`*` = 큐의 일부만 포함). 번역에서 `mapper.fragments()`는 이런 조각마다 `[n] 조각` 줄을 만들고,
같은 큐의 조각 번역들을 `merge_to_units`가 공백으로 이어 붙인다. 즉 **한 문장이 두 번의 LLM 호출로 나뉘어 번역**된다 — 번역 품질 저하의 직접 원인.

**smart (srt 모드, min 15 / max 40단어, window 2)** — 한 섹션 = 문서 전체. 큐 임베딩으로 인접 큐 사이의 점수를 매기고(낮을수록 자르기 좋음) 최저점에서 재귀 분할:

```
경계점수(큐 i | i+1):  6|7 0.74  7|8 0.08  8|9 0.72  9|10 0.86  10|11 0.85  11|12 0.63
                        12|13 0.07  13|14 0.73  16|17 0.71  17|18 0.08  18|19 0.56 ...
        (1.00 근처 = 같은 주제로 이어지는 곳, 0.07–0.08 = 주제가 바뀌는 곳)

결과 청크 (큐 경계에서만 자름 → 조각 없음)
  #0 큐 1–7   25단어  00:03:00,430 → 00:03:22,905   택시에서 인사
  #1 큐 8–10  27단어  00:03:28,876 → 00:03:40,854   아내감·사촌 이야기
  #2 큐 11–12 20단어  00:03:42,014 → 00:03:47,645
  #3 큐 13–16 38단어  00:03:52,566 → 00:04:10,880   "세상은 … 맛이 난다" 비유
  #4 큐 17–19 15단어  00:04:16,882 → 00:04:23,847
```

- 모든 청크는 **큐 경계에서만** 끝나므로 조각(`*`)이 없다. 청크마다 `t_start/t_end`가 정확히 정해진다.
- 이 발췌에는 문장이 다음 큐로 이어지는 경우가 없어 가산점이 쓰이지 않았다. 이어지는 경우의 규칙은 아래와 같다:

  ```
  큐 A: "I told you that if you"      큐 B: "went there you'd regret it."
  cue_continues(A, B, "en") = True  →  경계점수 += 1.0  →  가장 높은 점수라 사실상 자르지 않음
  (영어: 문장부호로 안 끝났거나 다음 큐가 소문자로 시작하면 이어짐 / 한국어: 연결어미 등으로 끝나면 이어짐 / 공통: "..."·"…"·","로 끝나면 이어짐)
  ```

- 현재 구현은 "유사도 임계값 미만이면 자른다"가 아니라 **max 토큰을 넘는 섹션을 가장 점수가 낮은 경계에서 재귀적으로 반으로 가르는** 방식이다(양쪽이 min 이상인 경계만 후보). 사용자가 설명한 "큐 단위로 묶은 뒤 유사도가 떨어지는 곳에서 분할"과 같은 방향이며, 이 문서는 구현을 바꾸지 않는다. 임계값 방식으로 바꾸는 것은 별도 실험으로 둔다.

### 5.4 청커별 파라미터 (현재 pipeline.py 값을 `config.py`로, fixed는 토큰 기준으로 변경)

| 데이터셋 | fixed | semantic | smart | 토큰 카운터 |
|---|---|---|---|---|
| vectara (pdf, en) | 목표 400토큰 → 환산 글자 수, overlap 0 | percentile 90, min 100 / max 400 | mode=pdf, lang=en, min 100 / max 400, window 2 | bge-m3 |
| allganize (pdf, ko) | 목표 400토큰 → 환산 글자 수(한국어 비율), overlap 0 | 같음 | mode=pdf, lang=ko, 같음 | bge-m3 |
| docx_corpus (docx, en) | 목표 400토큰 → 환산 글자 수, overlap 0 | 같음 | mode=docx, lang=en, 같음 | bge-m3 |
| srt (en↔ko) | 목표 400토큰 → 환산 글자 수(원문 언어별), overlap 0, `strip_whitespace=False` | percentile 90, min 128 / **max 512** | mode=srt, lang=원문 언어, min 128 / max 512, window 2 | Gemma |

- 위 값은 튜닝 전 **시작값**이다(`docs/design.md`). 튜닝은 8절 프로토콜로 한다.
- **srt semantic의 max는 기존 1024에서 512로 낮춘다.** 기존에는 semantic 1024 / smart 512로 서로 달랐는데, 청커 간 공정한 비교를 위해 같은 값으로 맞추는 의도적 변경이다(8절의 srt 후보 범위도 두 청커가 같다). 이 변경은 보고서에 기록한다.
- fixed의 기존 "500자"는 영어에서 대략 150~200토큰이어서(아래 측정 참고) semantic/smart의 상한 400토큰보다 훨씬 작았다. 그래서 fixed도 토큰 기준으로 맞춘다.
- allganize는 이전에 "docx 트랙(한국어 docx)"으로 돌던 데이터였고 지금은 PDF만 남았으므로 `mode=pdf`로 옮긴다. lang=ko에서는 `cue_continues`가 안 쓰이고(srt 전용) 문장 분리 `SENT_RE`는 영어 문장부호 기준이다(주석에 "한국어는 `kss`로 바꾸면 더 정확"이라고 적혀 있음 → 한계로 기록, 변경하지 않음).

#### 5.4.1 fixed 글자 수 환산 — 측정 방법

fixed는 글자 수로 자르는 청커라서, 다른 청커와 크기를 맞추려면 "목표 토큰 수에 해당하는 글자 수"가 필요하다. 이 비율은 언어·토크나이저·문서 성격(수식·표)에 따라 다르므로 **추정하지 않고 측정한다.**

1. **시점**: 전처리(2단계) 직후, 청킹(3단계) 전. `scripts/measure_ratio.py --dataset X`.
2. **표본**: 전처리된 JSON의 `to_text(doc)`. 문서가 100개 이하면 전부, 그보다 많으면 시드 고정 무작위 표본(최소 20개). 평문 전체(제목·표 포함)를 쓴다 — 실제 청킹 대상과 같다.
3. **계산**: 데이터셋의 토큰 카운터(5.4절 표의 마지막 열)로 표본 평문의 총 토큰 수를 센다.
   - `chars_per_token = Σ 글자 수 / Σ 토큰 수` (문서별 비율의 평균이 아니라 **합계 비율**)
   - 문서별 `chars/token`의 최솟값·중앙값·최댓값도 함께 기록한다.
4. **환산식**: `chunk_size_chars = round(target_tokens × chars_per_token)`, `chunk_overlap_chars = round(overlap_ratio × chunk_size_chars)`.
5. **저장**: `data/manifests/<dataset>.ratio.json`(매니페스트가 목록 형태라 별도 파일)의 `by_lang.<lang>`에 `{chars_per_token, n_docs, total_chars, total_tokens, per_doc_min/p50/max}`와 측정 시각·토크나이저를 기록한다. 코드에 숫자를 박지 않는다. 전처리를 다시 하면(3.7절) 비율도 다시 잰다. 측정 문서 수가 매니페스트 문서 수와 다르면 실패한다.
6. **언어별**: 영어(vectara, docx_corpus)와 한국어(allganize)는 따로 잰다. srt는 원문 언어마다(영→한이면 영어 자막, 한→영이면 한국어 자막) 따로 잰다.

**2단계 실측값 (2026-10-02, Docling 변환 평문 기준, 전체 문서)**

| 데이터셋 | 언어 | 토크나이저 | chars/token | 문서별 최소~최대 (중앙) | 목표 400토큰 ≈ |
|---|---|---|---|---|---|
| vectara | en | bge-m3 | 3.357 | 2.67 ~ 4.39 (3.46) | 1,343자 |
| docx_corpus | en | bge-m3 | 4.335 | 3.19 ~ 5.01 (4.39) | 1,734자 |
| allganize | ko | bge-m3 | 1.957 | 1.45 ~ 2.73 (1.88) | 783자 |
| srt | en | Gemma | 3.433 | 3.34 ~ 3.52 (3.43) | 1,373자 |
| srt | ko | Gemma | 1.494 | 1.43 ~ 1.56 (1.49) | 598자 |

같은 영어라도 논문(수식 많음) 3.36과 Word 문서 4.34가 크게 다르다. 즉 글자 수를 고정하면 데이터셋마다 청크 토큰 수가 달라지므로 데이터셋별 환산이 필요하다.

**참고: 계획 단계의 예비 측정값 (위 실측 이전, 비교용이며 Docling 변환 텍스트가 아니라 vectara 원본 섹션 텍스트·allganize 정답문·영어 자막 기준 — 구현 때 위 방법으로 다시 잰다)**

| 대상 | 토크나이저 | chars/token | 비고 |
|---|---|---|---|
| 영어 논문 (vectara 20편) | bge-m3 | 평균 2.55 (문서별 1.83~3.59) | 500자 ≈ 196토큰, 400토큰 ≈ 1,020자 |
| 영어 논문 (같은 20편) | Gemma | 평균 3.34 | |
| 한국어 (allganize 정답문) | bge-m3 / Gemma | 1.93 / 1.68 | |
| 영어 자막 (Deadpool) | Gemma | 3.84 | |

- 문서별 편차가 커서(수식이 많으면 낮아짐) 같은 글자 수여도 청크 토큰 수가 문서마다 ±30% 흔들린다. 이는 fixed의 특성이며, 그래서 6.2절에 **평균 청크 토큰 수**를 함께 보고한다.
- semantic/smart의 `max_tokens`는 **상한**이어서 실제 평균 청크는 상한보다 작다. fixed의 목표 토큰을 이 상한에 맞추는 것은 근사이며, 이 차이는 평균 청크 토큰 수·top-5 총 토큰 수로 투명하게 보인다.

### 5.5 `scripts/chunk_stats.py`

```
python scripts/chunk_stats.py --dataset vectara --chunker smart --max 400
```
청크 파일을 만들고(이미 있으면 재사용) 아래 통계를 `results/<dataset>/chunk_stats/<chunker>__<설정>.json`에 쓴다. 이 값이 "단순 분할이 문맥을 끊는가"의 정량 근거가 된다.
**1차 구현 지표는 아래로 한정한다.** 나머지는 14절로 미룬다.

| 지표 | 정의 |
|---|---|
| `n_chunks`, `tokens_{mean,p50,p95,max}` | 크기 분포(데이터셋 토큰 카운터 기준) |
| `over_max_rate`, `under_min_rate` | max 초과 / min 미만 청크 비율 (fixed는 max·min을 목표 토큰 기준으로 계산) |
| `partial_start_rate`, `partial_end_rate` | 블록 중간에서 시작/끝나는 청크 비율 |
| `sentence_cut_rate` | `partial_end=true`이면서 끝 글자가 문장 종결 부호(`. ! ? … ” ) ]` 등)가 아닌 청크 비율. 청커가 **문장 중간**을 잘랐는지를 센다. 제목·목록·표의 자연스러운 끝(`partial_end=false`)은 세지 않는다. pdf/docx 전용 |
| `table_split_rate` | 표가 둘 이상의 청크에 걸친 비율(표 단위) |
| `orphan_heading_rate` | 마지막 블록이 제목인 청크 비율 |
| `cues_split_ratio` (srt) | 2조각 이상으로 갈려 번역된 **큐**의 비율(전체 큐 대비) |

`sentence_cut_rate`의 한계: `et al.`, `Fig. 3`, `e.g.`, 소수점 뒤에서 끝나면 종결 부호로 잘못 판정한다(근사값). 한국어는 마침표 없는 문장이 있어 정확도가 낮을 수 있다. 보고서에 한계로 적는다.

---

## 6. 인덱싱·검색·평가 (③④)

### 6.1 인덱스 (`indexing/index.py`)

```python
def build_index(chunks, embeddings) -> InMemoryVectorStore       # 현재 retrieval.build_retriever와 동일
def save_index(store, dataset, chunker_tag)                       # InMemoryVectorStore.dump
def load_index(dataset, chunker_tag, embeddings) -> store | None  # 폴더가 있으면 load, 없으면 None
def get_retriever(store, k=5)                                     # store.as_retriever(search_kwargs={"k": k})
```
- 코퍼스 전체(데이터셋의 모든 문서)를 인덱스 하나에 넣고, 질문마다 top-5를 검색한다(현재 방식 유지 — "여러 문서 중 정답 문서를 찾는" 문제이려면 코퍼스 전체가 후보여야 함).
- 임베딩: `BAAI/bge-m3`, `normalize_embeddings=True`.
- **tune 실행에서도 인덱스는 전체 코퍼스로 만든다.** 질문만 tune/eval로 나눈다(tune 문서만으로 만들면 코퍼스가 작아져 eval보다 쉬워 점수를 비교할 수 없다).

### 6.2 지표 (`tasks/retrieval.py`)

**기존 지표 (식 그대로)**
- `hit@5`: top-5 청크 중 정답 *문서*의 청크가 하나라도 있으면 1.
- `MRR`: top-5 안에서 정답 문서 청크의 첫 순위의 역수.
- `F1`: top-5 중 정답 문서 청크들과 `answer`의 SQuAD word-F1 최댓값(정규화: 소문자·구두점 제거·공백 정리).

기존 `hit`은 **문서 단위**라서 청커 간 차이가 작게 나올 수 있다(정답 문서의 아무 청크나 걸리면 성공). 그래서 근거가 있는 데이터셋에는 아래 지표를 *추가*한다(기존 지표를 대체하지 않음).

**추가 지표**

| 지표 | 대상 | 정의 |
|---|---|---|
| `evidence_recall@5` | docx_corpus | 질문마다 **정답 문서의 top-5 청크를 합친 글자 범위가 근거 스팬 `[s,e)`을 덮는 비율**(0~1)을 구하고 질문 평균을 낸다. 같은 `to_text` 오프셋 기준이라 정확하다. (기존의 "겹침 30% 이상이면 1" 기준 지표는 청크 크기에 민감해서 삭제했다.) |
| `section_hit@5` | vectara | top-5 중 **정답 문서의 청크**에 대해, **청크 토큰 중 정답 섹션 텍스트에 포함된 비율**이 0.5 이상인 청크가 하나라도 있으면 1. "이 청크가 그 섹션에서 나온 것인가"를 판정한다. 섹션이 청크보다 훨씬 길어도 도달 가능하도록 방향을 청크→섹션으로 잡았다(7.1절). |
| `page_hit@5` | allganize | 청크의 `[page_start, page_end]`가 `gold_page`를 포함하면 1. **사용 전 확인**: Docling의 `page_no`(1부터 시작으로 알고 있음)와 `target_page_no`가 같은 기준인지 문서 2~3개로 직접 확인한 뒤에 쓴다. 여러 쪽에 걸친 큰 청크가 유리하다는 편향이 있다. |
| `by_qtype` | 전체 | `qtype`(extractive/abstractive/table)별 위 지표 |

**대표 지표 규칙.** 데이터셋마다 대표 지표 하나(docx=`evidence_recall@5`, vectara=`section_hit@5`, allganize=`page_hit@5`)와 `MRR`을 보고한다. 0/1 지표와 연속 지표를 섞어 평균 내지 않는다.
세 지표는 정의가 서로 달라 **데이터셋 간 직접 비교는 하지 않는다.** 튜닝 선택 기준도 대표 지표를 쓴다(8절).

**결과 표에 함께 적을 값 (크기 효과 해석용)**
- 청커별 **평균 청크 토큰 수**
- 질문별 **top-5 총 토큰 수**의 평균 (overlap이 있는 fixed는 중복 토큰도 그대로 센다)

`evidence_recall`, `page_hit`처럼 가져오는 글이 많을수록 유리해지는 지표를 해석하려면 이 두 값이 필요하다. 이 값이 청커 간에 크게 다르면 차이가 청킹 방식 때문인지 크기 때문인지 구분해서 서술한다.
문서 단위 지표는 청크 경계와 무관하게 비슷해질 수 있다는 점, 근거 단위 지표는 청크 경계에 민감하다는 점을 보고서에 함께 서술한다.

### 6.3 평가 흐름 (`scripts/run.py --task retrieval`)

```
manifest/qa 로드 → split 선택(tune|eval) → processed JSON 로드
  → chunks 파일 있으면 재사용, 없으면 chunk() 후 저장
  → index 폴더 있으면 load, 없으면 build 후 save (--force면 다시 만듦)
  → 질문마다 retriever.invoke(question) → 지표 계산
  → results/<dataset>/retrieval/<chunker>__<설정>.json 저장
```
- `--limit N`은 QA를 N개로 제한(문서는 전부 인덱싱) — 현재 의미 유지.
- 이미 결과가 있으면 건너뜀(이어하기), `--force`로 덮어쓰기.

---

## 7. QA 구축

### 7.1 vectara — 추천안

현재 상태: `queries/answers/qrels.json`은 질문 id로 묶이고, `qrels`는 `{doc_id, section_id}`를 준다.
현재 검색 평가 코드는 `doc_id`만 쓰고 `section_id`는 쓰지 않는다. `corpus/*.json`은 HF 원본의 논문별 **섹션 단위 텍스트**(`sections[].text`, `section_id`)다.

| 안 | 내용 | 평가 |
|---|---|---|
| A. 삭제 | `corpus/*.json` 제거 | 단순하지만 정답 *섹션* 정보를 못 쓴다. |
| B. 그대로 보관 | `raw/vectara_pdf/_meta/corpus/`에 두기만 | 원본 보존. 평가 개선은 없다. |
| **C. 보관 + 근거 추출 (추천)** | B + `qrels.section_id`로 정답 섹션 텍스트를 QA 레코드의 `evidence.text`에 저장 | 원본 보존, 문서 단위보다 **정밀한 근거 단위 지표**(`section_hit@5`)를 쓸 수 있다. |

C안 세부:
1. `data/raw/vectara_pdf/_meta/corpus/`에 100개 JSON을 보관한다(git 제외, HF 원본이므로 언제든 재다운로드 가능).
2. `scripts/build_qa.py --dataset vectara`가 `queries/answers/qrels`를 합치며 `section_id`에 해당하는 섹션 텍스트를 `evidence`에 넣는다.
3. 주의: HF의 섹션 텍스트와 우리 Docling 변환 텍스트는 **서로 다른 파서의 결과**라 글자 단위로 일치하지 않는다. 그래서 오프셋 일치가 아니라 *토큰 포함 비율*로 판정한다. 방향은 **청크 토큰 중 정답 섹션 텍스트에 포함된 비율 ≥ 0.5**(청크가 그 섹션에서 나온 것인지 판정)이고, 정답 문서의 청크만 대상으로 한다. 이 지표의 한계는 보고서에 적는다: 파서 차이, 표·수식 표기 차이, 섹션 두 개에 걸친 청크는 절반 미만으로 떨어져 불리함.
4. 기존 `hit@5/MRR/F1`은 식 그대로 계산한다(지표 정의의 연속성만 유지하며, 이전 실행 결과와 비교하지는 않는다).

질문 유형: `type`(extractive/abstractive)과 `source`(text/text-table)를 각각 보존한다. 이미지 질문은 이미 제외되어 있다.

### 7.2 allganize

- `documents.csv`의 `file_name`(로컬 PDF 이름)과 `orig_file_name`(원본 이름)을 이용해 `rag_evaluation_result.csv`의 `target_file_name` → `doc_id`를 연결한다(현재 `retrieval.load_docx_qa`와 같은 방식, 45/45 일치 확인됨).
- QA 필드: `question`, `answer`(=`target_answer`), `doc_id`, `gold_page`(=`target_page_no`), `qtype`(=`context_type`, 예: `table`, `image`).
- **`context_type=image` 질문은 제외한다**(확정). 우리는 이미지 처리를 하지 않으므로 vectara와 같은 원칙이다. 해당 질문은 `excluded: true`로 표시하고 `split`은 `null`로 둔다(3.3절). 현재 `context_type` 분포는 image 45 / paragraph 111 / table 42 / text 13이고, 제외 후 **166개**(표 질문 42개 포함)가 평가 대상이다. 이미지 질문이 가리키는 문서도 다른 질문 때문에 코퍼스에는 그대로 들어간다.
- CSV의 모델 답변·O/X 열(`alli_*`, `langchain_*` …)은 QA에 넣지 않는다(참고용으로 원본 CSV는 raw에 보관 가능).

### 7.3 docx_corpus — QA 생성 파이프라인 (`qa_build/gen_docx_qa.py`)

목표(확정): 문서 100개에서 **최종 300개 이상**의 검증된 질문(문서당 3~5개, 표가 있는 36개 문서는 표 질문 1개 추가 시도). 수동 검수 비율은 **10%**.
생성 모델은 프로젝트에서 이미 쓰는 로컬 Gemma(`TOKENIZER` 설정)이고 외부 API는 쓰지 않는다.

**공정성 원칙 (가장 중요)**
- 근거 구간을 *제목 섹션 단위로* 고르면 구조 기반 smart 청커에 유리한 평가가 된다. 따라서 근거는 **청커와 무관하게** 고른다:
  문서에서 무작위 시작 블록을 정하고, 문단 블록을 이어 붙여 약 250~400 토큰이 될 때까지 늘린다(제목은 넘나들 수 있음).
- 근거는 `to_text` 오프셋(`start,end`, `block_ids`)으로 저장하므로 어떤 청커로 잘라도 같은 기준으로 채점된다.
- 생성에 쓴 임베딩·검색기를 필터에 쓰지 않는다(검색기에 맞춘 질문이 되는 순환을 피함).

**단계**
1. **구간 샘플링**: 문서마다 시드 고정(42)으로 서로 겹치지 않는 구간 후보 8개를 뽑는다. 목차·참고문헌·서식 표(빈칸 위주) 구간은 제외.
2. **질문 생성**(Gemma, 구간당 1개): 지시문 요지 — *"이 구간만 읽으면 답할 수 있지만, 구간을 보지 않은 사람도 이해할 수 있게 구체적인 고유명사·숫자·용어를 넣어라. 'the passage', 'this document' 같은 표현 금지. 정답은 1~3문장."* 출력은 JSON(`question`, `answer`, `qtype`).
   - 유형: extractive(구간 안 문장으로 답) / abstractive(여러 문장 종합) / table(표 블록이 있는 구간, 표 값을 묻는 질문).
3. **자동 필터**
   - 정답 근거성: 정답의 내용어가 구간에 충분히 포함(단어 겹침 ≥ 임계값) + Gemma에게 "이 구간만으로 답할 수 있는가? O/X"를 묻는다. 둘 다 통과해야 한다.
   - 질문 단독 이해성: 정규식으로 'the document/this section/the above' 등 지시어 포함 질문 제거.
   - 문서 특이성: 질문의 내용어 중 **코퍼스 문서 빈도(DF)가 낮은 단어(≤3개 문서)가 1개 이상** 있어야 한다. (임베딩이 아닌 어휘 통계라서 특정 검색기에 편향되지 않는다.)
   - 중복 제거: 같은 문서 안에서 질문 임베딩 코사인 ≥ 0.9인 쌍은 하나만 남김.
4. **수동 검수**: 필터를 통과한 질문 중 **무작위 10%(약 30~50개)**를 사람이 확인해 통과율을 기록한다. 통과율이 낮으면 지시문·임계값을 고쳐 재생성한다. 검수한 질문은 `reviewed: true`로 표시한다.
5. **저장**: `data/qa/docx_corpus.jsonl`. `evidence.kind="span"`, `evidence={block_ids, start, end}`, `qtype`, `split`(8절). 근거 스팬이 `to_text` 오프셋이므로 **전처리를 다시 하면 근거를 다시 계산**해야 한다(3.7절).

**알려진 한계 (보고서에 서술)**
- 질문 생성·정답 판정·답변 생성이 모두 같은 Gemma 계열이면 자기 선호 편향이 있을 수 있다. 완화: 검수 표본, 어휘 기반 필터, 생성에 쓰지 않는 임베딩 모델로 평가.
- LLM이 만든 질문은 사람이 만든 vectara/allganize QA보다 쉬울 수 있다. 데이터셋별 결과를 합치지 않고 따로 보고한다.

**구현 결과와 계획 변경 (2026-10-06, docx_corpus QA 생성 완료)**
- 생성자는 Gemma가 아니라 **Claude(claude-sonnet-5-5, Claude Code 세션)**로 바꿨다: 평가용 LLM(Gemma)과 질문 생성자를 분리해 자기 선호 편향을 끊기 위함. 생성은 수동이라 재현하지 않고, 결과 파일(`data/qa/docx_corpus.jsonl`)을 고정해서 쓴다.
  지침은 `data/raw/docx_corpus/_meta/gen_prompt.md`, 생성 원본은 `_meta/gen/batch_*.json`에 남겼다.
- 순서: 전처리 확정 -> 샘플링(`scripts/sample_docx_spans.py`) -> 생성 -> 자동 필터(`qa_build/docx_filters.py`) -> `scripts/build_qa.py --dataset docx_corpus`.
  전처리를 바꾸면 근거 위치가 틀어지므로 `docx_candidates.stale_report/relocate`로 검사·재배치한다(3.7절).
- 근거 구간: 길이 150~600토큰 무작위(시드 42), 청커·제목과 무관. **끊김 검증**(문장 중간·번호 항목으로 시작하거나 콜론·접속어로 끝나면 앞뒤로 확장하고 상한을 넘으면 버림), 표는 통째로만, 목차·참고문헌·점선 목차 구간 제외.
- 전처리 수정: Docling이 표의 셀 문단을 표의 자식으로도 내보내 같은 내용이 표와 문단으로 두 번 나오는 문제를 고쳤다(`table_cell_dups`, docx 24개 문서에서 710블록 제거, 글자 손실 0).
- 필터(어휘 통계만): 근거 인용이 구간에 있음(support), 정답 단어의 구간 포함(grounding), 지시어 없음(deictic), 문서 특정성(specific: **희귀 단어 규칙을 TF-IDF 순위 10위 이내로 교체** - 규칙에 걸린 질문이 전부 정답 문서 1위였기 때문),
  8단어 이상 복사 없음(copy, 이름 구간은 예외), **근거 인용이 코퍼스 전체에서 한 번만 나옴(unique)**, 같은 문서 안 중복 없음. 판만 다른 규격·법령 문서가 따로 있어 같은 문장이 여러 문서에 있는 경우가 많았다.
- 결과: 후보 500개(문서 100개 x 5) -> 질문 417개(extractive 255 / abstractive 143 / table 19), 질문이 있는 문서 94개. 한계: 법령·규정 문서 비중이 높고(어휘 겹침이 큰 질문), 사람 검수 10%는 아직 하지 않았다. 질문 0개 문서 6개는 모두 이력서·법령 목록·다른 판과 중복인 문서다.
- 평가 지표: 근거가 `evidence.kind="span"`이라 `section_hit`가 아니라 `evidence_recall@5`(10절)로 채점한다. 아직 구현하지 않았다.

### 7.4 srt

QA가 없다. 번역 품질은 **참조 자막**(같은 영화의 다른 언어 자막)을 시간 겹침(Jaccard)으로 큐에 정렬해 채점한다(`timestamp_align.align_by_overlap`, 현재와 동일).
영화 쌍 목록은 `manifests/srt.json`이 가진다.

---

## 8. tune / eval 분할과 튜닝 프로토콜

**분할**
- 문서 단위로 20% tune / 80% eval. 같은 문서의 질문이 두 세트에 걸치지 않게 한다(누수 방지).
- 데이터셋 안에서 `type`(docx/vectara) 또는 `domain`(allganize)으로 층화, 시드 42. 결과를 `manifests/<dataset>.json`의 `split`과 `qa/*.jsonl`의 `split`에 저장하고 git에 포함한다. allganize는 이미지 질문 45개를 뺀 166문항 기준이며, 제외된 질문은 `split=null`이다.
- **인덱스는 항상 전체 코퍼스로 만들고 질문만 tune/eval로 나눈다**(6.1절).
- srt는 영화 5편 중 **1편을 tune**, 4편을 eval로 한다. tune 영화는 번역 비용이 가장 작은 *Insidious Chapter 2*(영어 자막 43KB로 최소)를 권장하며, 실제 번역 시간을 재 본 뒤 확정한다. 영화 수가 적어 tune 신뢰도가 낮은 점은 한계로 기록한다.

**튜닝 (`scripts/tune.py`)**
- tune 세트에서만 파라미터를 고른다. eval 세트는 최종 1회만 쓴다.
- **세 청커 모두 후보 9개**를 같은 예산으로 탐색한다(검색 데이터셋: vectara, allganize, docx_corpus):

| 청커 | 탐색 후보 (3 × 3 = 9) |
|---|---|
| fixed | 목표 토큰 ∈ {300, 400, 512} × overlap ∈ {0, 10%, 20%} (글자 수는 5.4.1절로 환산) |
| semantic | percentile ∈ {85, 90, 95} × max_tokens ∈ {300, 400, 512} |
| smart | max_tokens ∈ {300, 400, 512} × window ∈ {1, 2, 3} |

- **srt는 번역 비용 때문에 세 청커 모두 `max_tokens` 3개({300, 400, 512})만 탐색**한다. fixed는 목표 토큰 {300, 400, 512}, semantic·smart는 `max_tokens` {300, 400, 512}로 범위가 같다(시작값을 512로 맞춘 이유는 5.4절).
  - **srt의 fixed overlap은 0으로 고정한다.** overlap이 있으면 같은 큐가 두 청크에 들어가 번역이 두 번 이어 붙으므로(`merge_to_units`가 조각을 공백으로 연결) 텍스트가 중복된다.
  - 규모: 영화 1편 × 2방향 × (3청커 × 3값) = **번역 18회**. 구현 중 번역 시간을 재 보고, 너무 길면 범위를 줄인다(14절).
- 선택 기준: tune 세트의 **대표 지표**(6.2절)로 고르고, 동률이면 `MRR`, 그래도 같으면 청크 수가 적은 쪽을 택한다. srt는 chrF.
- 크기 효과 주의: `max_tokens`나 목표 토큰이 큰 설정이 `evidence_recall`·`page_hit`에서 유리할 수 있다. 그래서 청커 간 비교는 **(a) 각자 최적 설정끼리**와 **(b) 같은 목표 크기(400)끼리** 두 가지로 보고하고, 결과 표에는 평균 청크 토큰 수와 top-5 총 토큰 수를 함께 적는다.
- 모든 시도 결과를 `results/<dataset>/tune/…`에 저장해 보고서의 어블레이션 표로 쓴다.

---

## 9. 태스크

### 9.1 `tasks/qa.py` (구 `summary.py`)

- 로직 그대로: top-k 청크로 Gemma가 질문에 답하고(`max_new_tokens = 입력 토큰×2+100`), 정답 포함 여부를 Gemma가 O/X로 판정 → `coverage = O 비율`.
- 입력만 바뀐다: 검색기와 QA를 `data/qa`와 인덱스 캐시에서 받는다. 판정 응답이 O/X가 아니면 원문 앞 20자를 기록하고 집계에서는 X로 센다(현재와 동일).
- **프롬프트 언어는 데이터셋 언어(`DATASETS[...]["lang"]`)를 따른다**(확정). `ko`(allganize)는 기존 한국어 프롬프트를 그대로 쓰고, `en`(vectara, docx_corpus)은 같은 구조의 영어 프롬프트(답변용·정답 판정용)를 쓴다. 두 벌은 `config.py`에 함께 보관한다. 판정은 여전히 `O`/`X` 한 글자로 받는다. srt는 이 태스크를 쓰지 않는다.

### 9.2 `tasks/translate_srt.py`

- 합치는 파일: `srt/mapper.py`(fragments, build_prompt, parse_marked, merge_to_units, write_srt), `srt/timestamp_align.py`, `srt/subtitle_translate.py`(방향별 프롬프트·few-shot, 배치 번역, chrF/BLEU/BERTScore, 타임스탬프 무결성 검사), `comet_score.py`.
- 흐름(현재와 동일):
  1. `to_lc_document(doc)`로 `Document(page_content, metadata["units"])` 생성
  2. 청커로 `split_documents([doc])` → 청크마다 겹치는 큐 조각을 `[n] 조각` 줄로 만들어 프롬프트 구성
  3. 배치(8청크)로 Gemma 번역 → `[n]` 번호로 파싱 → 큐 단위로 병합
  4. `write_srt`: **타임스탬프는 원본 그대로**, 본문만 번역문으로 교체
  5. `check_timestamp_integrity`: 출력 SRT의 큐 개수·시각이 원본과 1e-6초 안에서 일치하는지 확인
  6. 참조 자막과 시간 겹침으로 정렬해 큐 단위 chrF·BLEU·BERTScore 평균
  7. **층별 분석(추가)**: 각 큐에 `n_fragments`(몇 조각으로 번역됐는지)를 기록하고, 점수를 *갈린 큐(n_fragments ≥ 2)* 와 *안 갈린 큐*로 나눠 따로 평균한다. 빈 번역 큐 수(`empty_units`)도 센다.
     → "단순 글자 수 분할은 큐를 가르고, 갈린 큐에서 번역 품질이 떨어진다 / smart는 큐 경계에서만 자르므로 갈린 큐가 0개"를 같은 조건에서 직접 보이는 근거가 된다.
- 방향: ko→en, en→ko 양쪽. 청커는 방향마다(`lang`이 달라 `cue_continues` 규칙이 다름) 다시 만든다.
- srt의 fixed는 **overlap 0 고정**이다(8절). overlap이 있으면 같은 큐가 두 청크에 들어가 번역문이 중복된다.
- `--fake-llm`: 모델 가중치 없이 청킹·매핑·타임스탬프 보존만 검증(현재 기능 유지).
- 결과 파일: `results/srt/translate/<chunker>__<설정>.json` + 번역 SRT 출력. 현재의 `자막_번역_비교결과.txt` 요약 리포트도 이어서 생성.

---

## 10. 실행 스크립트 (CLI)

모두 `.venv311/bin/python`으로 실행한다. `scripts/*`는 로직 없이 `src`를 호출한다.

```
# ① 전처리 (한 번만). --force는 JSON을 다시 만들고 chunks/·index/ 삭제, results/ → results/_stale/ (3.7절)
python scripts/preprocess.py --dataset vectara
python scripts/preprocess.py --dataset docx_corpus
python scripts/preprocess.py --dataset srt

# fixed 글자 수 환산용 비율 측정 (전처리 직후, 5.4.1절)
python scripts/measure_ratio.py --dataset vectara

# QA 구축
python scripts/build_qa.py --dataset vectara        # queries/answers/qrels 합치기 + 근거 추출
python scripts/build_qa.py --dataset allganize      # CSV → jsonl
python scripts/build_qa.py --dataset docx_corpus    # Gemma로 생성 (7.3절)

# ② 청크 통계 (실험 전 sanity + 보고서 정량 근거)
python scripts/chunk_stats.py --dataset vectara --chunker smart --max 400

# 튜닝 (tune 세트)
python scripts/tune.py --dataset vectara --chunker smart

# ③④ 평가 (eval 세트)
python scripts/run.py --dataset vectara --task retrieval --chunker smart --max 400
python scripts/run.py --dataset vectara --task qa        --chunker all
python scripts/run.py --dataset srt     --task translate --chunker all --fake-llm
```
공통 옵션: `--split {tune,eval,all}`, `--limit N`, `--force`, `--chunker {fixed,semantic,smart,all}`, `--min/--max`(smart·semantic 토큰), `--target-tokens`·`--overlap`(fixed), `--percentile`, `--window`.
`--force`의 의미는 스크립트마다 다르다: `preprocess`는 3.7절 규칙, `run`/`tune`은 해당 결과 파일 덮어쓰기, 인덱스는 다시 만들기.

---

## 11. `common/config.py` 내용

- 경로: `ROOT`, `RAW_DIR`, `PROCESSED_DIR`, `CHUNKS_DIR`, `INDEX_DIR`, `QA_DIR`, `MANIFEST_DIR`, `RESULTS_DIR`.
- 모델: `EMBED_MODEL = "BAAI/bge-m3"`, `TOKENIZER = "google/gemma-4-12B-it-qat-q4_0-unquantized"`(현재 값 유지).
- `DATASETS` 표(구현 시 단일 진실 공급원):

```python
DATASETS = {
  "vectara":     dict(fmt="pdf",  lang="en", raw="vectara_pdf",  counter="bge-m3", tasks=["retrieval","qa"],
                      chunkers=dict(fixed=dict(target_tokens=400, overlap_ratio=0.0),   # chunk_size는 manifest의 측정 비율로 계산
                                    semantic=dict(percentile=90, min_tokens=100, max_tokens=400),
                                    smart=dict(mode="pdf", min_tokens=100, max_tokens=400, window=2))),
  "allganize":   dict(fmt="pdf",  lang="ko", raw="allganize_pdf", counter="bge-m3", tasks=["retrieval","qa"], ...),
  "docx_corpus": dict(fmt="docx", lang="en", raw="docx_corpus",   counter="bge-m3", tasks=["retrieval","qa"], ...),
  "srt":         dict(fmt="srt",  lang=("en","ko"), raw=("srt_eng","srt_kor"), counter="gemma", tasks=["translate"],
                      chunkers=dict(fixed=dict(target_tokens=400, overlap_ratio=0.0, strip_whitespace=False),
                                    semantic=dict(percentile=90, min_tokens=128, max_tokens=512),   # 기존 1024 → 512 (smart와 맞춤, 5.4절)
                                    smart=dict(mode="srt", min_tokens=128, max_tokens=512, window=2))),
}
```
- 영화 쌍(`SUBTITLE_DATASETS`)은 코드가 아니라 `manifests/srt.json`으로 이동.
- fixed의 `chunk_size`(글자 수)는 `config.py`에 없다. `manifests/<dataset>.ratio.json`(5.4.1절 측정값)에서 `chunking.ratio.fixed_chunk_size()`로 계산한다.
- `qa_prompt`(ko/en 두 벌)도 `config.py`에 둔다(9.1절).

---

### 11.1 실행 환경 (재현용 기록)

모든 작업은 `.venv311`만 쓰고 서버 전역 환경(시스템 파이썬, `/root` 아래 전역 경로)은 건드리지 않는다.

| 항목 | 값 |
|---|---|
| 실행 파이썬 | `.venv311/bin/python` → 프로젝트 안 `.python/cpython-3.11-linux-x86_64-gnu/bin/python3.11` (3.11.17) |
| 복구 경위 | 원래 링크 대상인 `/root/.local/share/uv/...`가 서버에서 사라져 링크가 끊겨 있었다. uv를 프로젝트 안 `.tools/uvpkg/`에 설치하고 `UV_PYTHON_INSTALL_DIR`·`UV_CACHE_DIR`을 프로젝트 폴더로 지정해 파이썬 3.11을 `.python/`에 받은 뒤, 링크와 `.venv311/pyvenv.cfg`의 `home`만 고쳤다. `.venv-eval`은 같은 이유로 아직 끊겨 있다. |
| 주요 패키지 | torch 2.7.1+cu118, transformers 5.16.1, sentence-transformers 6.0.0, bitsandbytes 0.50.2, accelerate 1.14.0, langchain-experimental 0.4.2, docling 2.127.0 |
| GPU | NVIDIA RTX A5000 24GB (CUDA 11.8), CPU 32코어, RAM 188GB |
| git 제외 | `.venv311/`, `.python/`, `.tools/`, `.home/`(홈 디렉터리로 쓰는 프로젝트 내부 폴더) |

**`env.sh`** (실행 전에 `source env.sh`):

```bash
export HF_HOME=/root/jupyter/LLM_chunking/.home/.cache/huggingface   # 모델 캐시를 프로젝트 안으로 고정
export HF_HUB_OFFLINE=1                                              # 실험 중 네트워크 접근 차단(재현성·속도). 모델을 새로 받을 때만 0으로
export TOKENIZERS_PARALLELISM=false
# 이 환경의 파이썬: .venv311/bin/python (시스템 python3 사용 금지)
```

### 11.2 모델 준비 (2026-10-02 완료)

두 모델 모두 공개 모델이라 인증이 필요 없다. 캐시 위치는 `HF_HOME` 아래 `hub/`이다.

| 모델 | 용도 | 크기 | 스냅샷 |
|---|---|---|---|
| `BAAI/bge-m3` | 임베딩(semantic·smart 청킹, 검색, 중복 제거), 토큰 카운터(pdf/docx) | 2.2GB | `5617a9f6…` |
| `google/gemma-4-12B-it-qat-q4_0-unquantized` | QA 답변·판정, QA 생성, srt 번역, 토큰 카운터(srt) | 23GB (`model.safetensors`) | `b6ed8627…` |

- 기타 캐시: Docling 레이아웃·표 모델 `docling-project/docling-layout-heron`, `docling-project/docling-models`(약 0.5GB, 전처리 시 사용).
- **Gemma는 4bit(nf4)로 올린다.** 24GB GPU에는 bf16 전체가 올라가지 않으므로 현재 `llm.load_llm`이 하는 대로 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16)`로 로드하고 `Gemma4UnifiedForConditionalGeneration`을 쓴다. 이 로직은 `common/models.py`로 그대로 옮긴다(수정 없음).
- 확인 결과(오프라인 모드): bge-m3 임베딩 차원 1024, Gemma 로드 약 48초·GPU 메모리 8.2GB, 짧은 생성 성공. 로드 후 GPU는 모두 반환됨.
- 한 프로세스에서 bge-m3와 Gemma를 동시에 올려도 메모리상 문제 없지만(약 8.2GB + 약 2.5GB), 청킹 단계(임베딩만)와 번역·QA 단계(Gemma)를 분리해 실행하는 것이 안전하다.
- 다시 받는 방법: `source env.sh` 후 `HF_HUB_OFFLINE=0`으로 `huggingface_hub.snapshot_download(...)`를 `.venv311/bin/python`으로 실행(스크립트는 `scripts/download_models.py`로 제공 예정).

---

## 12. 마이그레이션 단계와 검증

기존 코드로 만든 출력(청크, 지표, 번역 결과)은 **기준으로 쓰지 않는다.** 이 구조로 완전히 전환하므로, 새 코드가 스스로 올바른지(자체 일관성)만 확인한다.
기존 `results/`의 이전 실행 결과는 새 구조로 옮기지 않고 `legacy/results_v2/`로 보관한다(삭제하지 않음).
각 단계는 아래 테스트를 통과해야 다음으로 간다. `legacy/`의 코드는 알고리즘을 옮길 때 참고만 한다.
**데이터 이동과 `src/ → legacy/` 이동은 맨 앞(1단계)에서 한 번에 끝낸다.** 새 `src/`는 빈 상태에서 작성한다.

| 단계 | 작업 | 통과 기준 (자체 일관성 테스트) |
|---|---|---|
| 1 | **이동 + 뼈대.** ① 이동 전 `data/` 파일 목록·개수(크기 포함)를 `docs/data_inventory_before_move.txt`에 기록 ② 데이터 이동(2.1절, 2026-10-02 폴더 이름 기준), `selected.json`의 `path` 재작성 ③ `src/` → `legacy/`, `results/` → `legacy/results_v2/` ④ `.gitignore` 정리(옛 폴더 이름 규칙 제거, `results/**` 새 규칙)와 `.git/info/exclude`와의 상호작용 확인(2.2절) ⑤ 폴더 뼈대, `config.py`, `docjson.py`(읽기/쓰기/렌더) | 이동 전후 파일 개수·크기 일치. `git check-ignore -v`로 `data/qa`, `data/manifests` 추적 가능 여부 확인. import 성공. 렌더 규칙 단위 테스트(4.1절 불변식 ①~④) |
| 2 | **전처리**: `*_prep.py` → JSON 생성 (각 포맷 소수 문서로 먼저 확인 후 전체) + `measure_ratio.py`로 fixed 글자/토큰 비율 측정·저장 | 블록이 비어 있지 않음, 제목 수준이 1부터 연속, `page` 범위가 `n_pages` 안, srt는 큐 개수 = 원본 큐 − 버려진 빈 큐, 변환 실패 문서 0건 또는 목록으로 기록. 비율이 `manifests/<dataset>.ratio.json`에 저장됨. **(2026-10-02 완료, 4.6절)** |
| 3 | **청킹**: `splitters.py`, `smart.py`, `attach_meta` | ① 모든 청크 `text == to_text(doc)[start:end]` (리터럴 슬라이스) ② 청크가 문서 순서대로이고, **overlap 0 설정에서는 서로 겹치지 않음**(overlap > 0이면 인접 청크의 겹침이 설정값 이내) ③ `block_ids`·`partial_*`·`page`·`t_*`가 오프셋과 일치 ④ 5.3절 예시 A·B의 기대 경계와 일치 ⑤ 같은 입력으로 두 번 돌려 결과 동일 |
| 4 | **인덱스·검색**: 캐시 포함 | 폴더에서 load한 인덱스와 새로 만든 인덱스의 top-5가 같음. 지표 단위 테스트(작은 가짜 코퍼스에 손으로 계산한 hit/MRR/F1/`evidence_recall`/`section_hit`). 문서 단위 지표는 만점 상황(질문 = 청크 그대로)에서 1.0 |
| 5 | **srt 번역 경로**: `to_lc_document` → mapper/번역 코드 | `--fake-llm`에서 `check_timestamp_integrity` OK. 번역을 원문 그대로 돌려줄 때 **공백을 제거하고 비교하면 모든 큐가 원문과 일치**하고, **공백을 포함한 비교에서 불일치하는 큐의 집합 ⊆ 청커가 가른 큐의 집합**(단어 중간에서 갈리면 조각 사이에 공백이 끼어 달라진다. 단어 경계에서 갈린 큐는 원문과 똑같이 복원되므로 불일치하지 않을 수 있다). `cues_split_ratio`는 불일치 큐가 아니라 **조각난 큐**를 센다 |
| 6 | `scripts/*` 작성, README 절차 점검 | 새 스크립트만으로 전 과정이 처음부터 재현됨(전처리 → 비율 측정 → 청킹 → 인덱스 → 평가) |
| 7 | QA 정리·생성(7절), 분할(8절), `chunk_stats`, 튜닝, 문서(`README`, `CLAUDE.md`, `docs/design.md` 경로) 갱신 | QA 문서 대응 100%, 분할에 문서 누수 0건, 전처리 `--force` 후 하위 산출물 정리 동작(3.7절) 확인, README 절차대로 재현 |

롤백: 단계마다 커밋(커밋은 사용자가 요청할 때만 한다). `legacy/`는 삭제하지 않는다. 1단계의 이동 전 파일 목록·개수 기록으로 이동이 정확했는지 대조한다.

---

## 13. 리스크와 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| Docling 항목을 블록으로 나눌 때 목록·캡션·수식·표를 잘못 다룸 | 표가 쪼개지거나 문단이 합쳐져 smart 구조 인식이 흐려짐 | 2단계 전처리 점검(표·제목 개수가 Docling 원본과 같은지), 이상한 문서는 목록으로 기록 |
| PDF 100편 재변환 시간·GPU 메모리 | 일정 지연, OOM | 변환 결과 JSON 캐시, 문서 단위 재시작, 서버의 cuDNN 우회 유지, OOM·트러블슈팅은 기록(보고서 필수 항목) |
| 임베딩 비결정성(배치·GPU)으로 실행마다 지표가 미세하게 달라짐 | 청커 간 작은 차이를 노이즈로 오판 | 같은 배치 크기·장치 고정(반복 실행 평균·신뢰구간은 14절 "나중에") |
| docx_corpus QA 품질 | 평가 신뢰도 | 7.3의 필터·수동 검수·한계 서술 |
| 구조 기반 근거 → smart 유리 | 비교의 공정성 훼손 | 근거를 청커와 무관한 구간으로 샘플링, 데이터셋별 결과 분리 보고 |
| `data/raw` 이동 중 실수 | 원본 손실 | 이동 전 파일 목록·개수(크기 포함) 기록 후 이동·대조(12절 1단계). 이동은 `mv`만 쓰고 삭제하지 않는다 |
| 전처리를 다시 한 뒤 낡은 청크·인덱스·결과가 재사용됨 | 오프셋 불일치, 조용히 틀린 결과 | `preprocess --force`가 하위 산출물을 정리(3.7절). 청크 로드 시 슬라이스 검사(선택) |
| `.git/info/exclude`의 `data/` 규칙 때문에 `data/qa`, `data/manifests`가 추적되지 않음 | QA·분할 정보 유실 가능 | 1단계에서 `git check-ignore -v`로 확인하고 예외 규칙 또는 `git add -f`(2.2절) |
| allganize의 한국어 문장 분리가 영어 구두점 기준 | smart/semantic 한국어 품질 저하 | 변경하지 않고 한계로 서술(`kss` 도입은 후속) |
| 현재 `docs/design.md`·`src/docx_track`이 사라진 allganize docx 경로를 가정 | 새 구조에서 혼동 | 12단계 7에서 문서 갱신, `legacy/`에만 남김 |

### 알려진 청킹 동작(변경하지 않고 측정만 하는 것)
1. smart: max를 조금 넘는 섹션이 유효 후보 없이 최저점에서 잘려 제목만 남는 청크가 생길 수 있음 (`orphan_heading_rate`).
2. smart: 섹션 경계를 넘어 작은 청크를 합치지 않음 → min 미만 청크가 남음 (`under_min_rate`).
3. smart: 표가 max보다 크면 쪼개지 않아 max 초과 청크가 생김 (`over_max_rate`).
4. semantic: 표를 모르고 줄 단위로 "문장"을 만들어 표가 갈릴 수 있음 (`table_split_rate`).
5. fixed: 단어·문장·표·큐 중간에서 자름 (`partial_*_rate`, `sentence_cut_rate`, srt는 `cues_split_ratio`).

---

## 14. 나중에 / 구현 중 확인할 것

**나중에 구현 (이번 범위 아님)**
- `heading_path`를 청크 앞에 붙이는 "제목 경로 접두사" 옵션: JSON과 청크 메타데이터에는 이미 값이 있으므로 `chunk(..., prefix_heading_path=False)` 같은 스위치 하나로 추가 가능. 비교 조건이 바뀌므로 **어블레이션 항목**으로만 다룬다(모든 청커에 동일 적용).
- 같은 설정의 **반복 실행 평균·표준편차**(임베딩 비결정성 확인). 재실행 없이 질문 단위 **paired bootstrap 신뢰구간**으로 청커 간 차이의 유의성을 보는 방법도 후보.
- `chunk_stats` 보조 지표: 청크 단위 `cue_split_rate`(조각난 큐가 있는 청크 비율), `cues_split_count`(갈린 큐 개수; 비율은 1차에 포함).
- srt smart를 "유사도 임계값 미만이면 분할" 방식으로 바꾸는 실험.
- 한국어 문장 분리(`kss`) 도입.
- 청크 겹침(overlap)의 단독 어블레이션(검색 데이터셋의 fixed 후보에는 이미 포함).
- srt 튜닝 범위가 번역 시간상 너무 길 때의 축소안.

**확정된 결정 (2026-10-02)**
1. allganize `context_type=image` 질문은 eval에서 제외한다(`excluded: true`) → 7.2절.
2. `tasks/qa.py` 프롬프트 언어는 데이터셋 언어를 따른다 → 9.1절.
3. docx_corpus QA는 최종 300개 이상, 수동 검수 10% → 7.3절.
4. srt tune은 영화 1편(권장: *Insidious Chapter 2*, 번역 시간 측정 후 확정) → 8절.
5. 인덱스 캐시는 `InMemoryVectorStore.dump/load`(지원 확인 완료), 해시 없이 "폴더 있으면 load / `--force`면 재생성" → 3.5절.

**구현 중에 확인할 것**
- Docling `page_no`와 allganize `target_page_no`의 기준이 같은지 문서 2~3개로 확인(6.2절 `page_hit@5`).
- `.git/info/exclude`의 `data/` 규칙과 `data/qa`·`data/manifests` 추적의 상호작용(2.2절).
- srt 번역 1회에 걸리는 실제 시간 → 튜닝 범위(8절) 확정.

---

## 15. 구현 체크리스트 (작업 순서 요약)

- [x] 0. 실행 환경(`.venv311` 파이썬 3.11 복구)·모델(bge-m3, Gemma 4bit) 준비 완료 (11.1·11.2절)
- [x] 1. 이동 전 `data/` 목록·개수 기록 → 데이터 이동(2.1절) → `src/` → `legacy/`(`results/`는 비어 있어 이동 없음) → `.gitignore` 정리·추적 확인 (2026-10-02 완료)
- [x] 1. 폴더 뼈대 + `env.sh` + `config.py` + `docjson.py`(render/to_text/block_spans/to_lc_document) + 단위 테스트 20개 통과 (2026-10-02 완료)
- [x] 2. `pdf_prep.py` → vectara 5편 검증 → 100편 변환 (allganize 45편 포함, 2026-10-02 완료)
- [x] 2. `docx_prep.py` → 5개 검증 → 100개 변환 (완료)
- [x] 2. `srt_prep.py` → 5편×2언어 (완료)
- [x] 2. `measure_ratio.py` → 데이터셋별 글자/토큰 비율을 `manifests/<dataset>.ratio.json`에 저장 (완료)
- [ ] 3. `splitters.py` + `smart.py` 이동, `attach_meta`, 자체 일관성 테스트(12절 3단계)
- [ ] 3. `chunk_stats.py`
- [ ] 4. `index.py`(dump/load) + `tasks/retrieval.py`(기존 지표 → 추가 지표)
- [ ] 5. `tasks/translate_srt.py` 통합, `--fake-llm` 검증
- [ ] 6. `scripts/*` 작성, README 절차 점검
- [ ] 7. `build_qa.py`(vectara/allganize) → docx_corpus QA 생성·검수 → 분할 저장
- [ ] 7. `tune.py` → eval 실행 → 결과 표
- [ ] 7. `README.md`, `CLAUDE.md`, `docs/design.md` 갱신 (데이터셋 개수·경로·실행 절차)

---

## 개정 이력

**v3.1 (2026-10-02)** — 수정 1~7과 확정 결정 반영

| 절 | 변경 |
|---|---|
| 머리말, 0 | v3.1 표시, 확정 결정 표 확장, allganize QA 166개 |
| 1 | 원칙 5에서 입력 JSON 해시 삭제, 3.7절 참조 |
| 2, 2.1, 2.2 | 현재 폴더 이름 기준 이동표, 모든 이동을 1단계로, `_meta/`·`ratio.py`·`measure_ratio.py` 추가, git 정책을 현재 상태(`.git/info/exclude`)에 맞춤 |
| 3.2, 3.3 | fixed 태그 `t400` 방식, `params`에 환산값, QA `excluded` 필드 |
| 3.5~3.7 | 인덱스 캐시 단순화, 3.7(전처리 재실행 시 캐시 정리 규칙) 신설 |
| 4.5, 5.1, 5.3 | `preprocess --force` 의미, fixed 글자 수 계산, 절 참조 정정 |
| 5.4, 5.4.1 | fixed 목표 토큰 기준, srt semantic max 512, 글자/토큰 비율 측정 방법 신설 |
| 5.5 | `chunk_stats` 1차 지표 한정, `sentence_cut_rate` 정의 |
| 6.1~6.3 | `load_index` 시그니처, 지표 3종 재정의(`evidence_recall@5`, `section_hit@5` 방향 반전, `page_hit@5` 확인 항목), 대표 지표 규칙, 청크 토큰 보고 |
| 7.1~7.3 | containment 방향, 이미지 질문 제외 확정, docx QA 확정 |
| 8 | 후보 9개씩, srt 규칙(overlap 0, max_tokens 3개), 선택 기준·크기 효과 보고, srt tune 영화 |
| 9.1, 9.2 | 프롬프트 언어 분기, srt fixed overlap 0 |
| 10, 11 | CLI 옵션, `DATASETS` 값 |
| 12 | 단계 재배열(1단계에 이동), 테스트 기준 수정 |
| 13, 14, 15 | 리스크 추가·수정, "나중에/확정/확인할 것" 재구성, 체크리스트 교체 |

**v3.2 (2026-10-02)** — 1~2단계 구현 결과 반영

| 절 | 변경 |
|---|---|
| 2.1, 2.2 | 1단계 완료 반영(이동 결과, `.gitignore` 규칙 확정), allganize·srt 매니페스트는 2단계에서 생성 |
| 3.1, 4.1 | 목록 항목은 항목마다 별도 조각, `footnote` 유형(맨 끝), 표 캡션은 별도 문단 블록(`role`/`for`) |
| 4.2, 4.3 | 2단계 확정 규칙(각주 A안, 캡션 중복 제거, compact 표, docx 제목 수준 이동) |
| **4.6 신설** | 2단계 실행 결과(문서·블록 수, 경고, 보존율, vectara 표 개수 정정) |
| 5.4.1 | 비율 저장 위치를 `<dataset>.ratio.json`으로, 실측값 표 추가 |
| 11, 12, 15 | 참조 정리, 2단계 완료 표시 |

**v3.3 (2026-10-03)** — PDF 정제 구현·적용

| 절 | 변경 |
|---|---|
| 4.2 | PDF 정제(PyMuPDF `clean`) 규칙·근거·옵션(`--no-pdf-clean`)·`delta_chars`, `pymupdf` 의존성 |
| 4.6 | vectara·allganize 재변환 결과(블록 수, 글자 수 변화, 누락 줄 복구, 중복 제거 확인) |
| 5.4.1 | vectara 글자/토큰 비율 갱신(3.357) |

**v3.4 (2026-10-06)** — semantic·smart를 공유 엔진으로 통합

| 절 | 변경 |
|---|---|
| 5 (머리 블록) | semantic = 엔진 + 규칙 off, smart = 엔진 + 규칙 on. 규칙 4개와 정책 결정(표 분할 금지, 제목 앞 컷도 min 미만이면 합침). 5.1·5.3·5.4의 옛 semantic/smart 설명은 v3.3 기록으로 남김 |
| 코드 | `semantic_core.py`·`units.py` 신설, `smart.py` → `legacy_smart.py`, `splitters.py`에서 `SemanticSpanner`·`recover_spans`·후처리 함수 삭제, 청크 파일 태그 규칙 변경 |
