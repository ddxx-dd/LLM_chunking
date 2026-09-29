# 스마트 청킹 비교 실험 계획 (프로젝트 2-1)

> 기준 레포: `ddxx-dd/LLM_chunking` · 작성일 2026-09-24, 검토/수정 2026-09-24
> 목표: **LangChain 글자수 분할 vs 의미 분할 vs 내가 만든 스마트 청커**를
> SRT · docx · pdf · LongBench 네 데이터셋에서 같은 조건으로 비교한다.
>
> **진행 순서(확정): docx_track → pdf_track(로더 보강 먼저) → SRT(한 줄 추가) → LongBench.**
> docx_track이 이미 100% 검증된 유일한 트랙이라 리스크가 제일 적어서 먼저 붙인다.
> 아래 각 절의 수정 사항은 실제 코드(`srt/loader.py`, `pdf_track/loader.py`,
> `docx_track/*`, `data/` 실제 파일)를 대조 확인해서 반영한 것.

---

## 1. 한눈에 보기

| 데이터셋 | 형식 | 청킹 후 하는 일 | 평가 방법 (원본 벤치마크 방식 그대로) |
|---|---|---|---|
| OpenSubtitles | SRT (EN↔KO) | 청크 단위 번역 → 큐 타임스탬프에 다시 매핑 | chrF, BLEU, BERTScore (지금 코드 그대로) + 타임스탬프 보존 |
| Allganize | docx (한국어) | 검색 → Gemma 답변 | 검색 적중(문서·페이지) + 답변 O/X 판정 (Allganize 방식) |
| Vectara Open RAG Bench | pdf (영어 논문) | 검색 → Gemma 답변 | 검색 적중(섹션) + MRR |
| LongBench v1 | 긴 텍스트 (영어) | 검색 → Gemma 답변 | LongBench 공식 지표 (QA F1) |

비교하는 청커는 딱 3개입니다.

| 이름 | 구현 | 설명 |
|---|---|---|
| `fixed` | LangChain `CharacterTextSplitter(separator="")` | 단순 글자수 분할 |
| `semantic` | LangChain `SemanticChunker` (SRT는 기존 `SemanticTextSplitter`) | 인접 문장 임베딩 유사도로 분할 |
| `smart` | `SmartTextSplitter` (새로 작성, 첨부 파일) | 구조 우선 → 의미 분할 → 크기 조절 |

> SRT만 기존 `SemanticTextSplitter`를 쓰는 이유: LangChain `SemanticChunker`는 문장을
> `" ".join()`으로 다시 붙여서 원문 위치(offset)가 깨지고, 그러면 타임스탬프 매핑이 안 됩니다.
> 알고리즘은 같습니다(percentile 임계값). 이미 레포에 있는 것을 그대로 씁니다.

---

## 2. 스마트 청커 알고리즘 (3단계)

참고: 기존 연구에서 반복해서 나오는 결론이 있습니다.
**"의미 분할이 항상 좋은 건 아니다"**. 의미 분할은 너무 작은 조각을 만들고,
문서 구조(제목, 표)를 무시하기 때문입니다. 그래서 스마트 청커는 이 두 약점만 고칩니다.

```
① 구조로 먼저 자른다
   - docx/pdf: 제목(#, "3.1 Method")이 나오면 새 섹션. 표는 절대 안 쪼갬
   - LongBench: "Passage N:" 마다 새 섹션
   - SRT: 큐 하나 = 쪼갤 수 없는 조각 (큐 중간은 절대 안 자름)

② 섹션이 max_tokens보다 크면, 의미가 가장 많이 바뀌는 곳에서 반으로 자른다
   - 조각 i와 i+1 사이 점수 = (왼쪽 2문장 평균 임베딩) · (오른쪽 2문장 평균 임베딩)
   - 점수가 가장 낮은 곳에서 자르고, 양쪽이 max_tokens 이하가 될 때까지 반복
   - SRT: 문장이 다음 큐로 이어지면("I was thinking that" → "maybe we could go")
     그 사이는 자르지 않도록 점수 +1

③ min_tokens보다 작은 청크는 다음 청크와 합친다
```

**기존 SemanticChunker와 다른 점 3가지**

1. 제목·표·큐 같은 **구조를 먼저 지킨다** → 표가 반으로 잘리지 않음
2. 인접 문장 1쌍이 아니라 **좌우 2문장 평균**으로 비교 → 문장 하나 튀는 것에 덜 흔들림
3. **토큰 수로 최소·최대 크기를 강제** → 43토큰짜리 자투리나 2000토큰짜리 덩어리가 안 생김
   (RFP의 "토크나이저로 엄격한 토큰 카운팅·로드 밸런싱"이 바로 이 부분)

**`smart_chunker.py` 검토 중 발견해서 고친 버그 2개** (코드에 반영 완료):
1. `merge_small()`이 **문서(정확히는 섹션) 맨 마지막 청크**는 "다음 청크"가 없어서 한 번도
   합쳐지는지 검사되지 않던 문제 → 마지막 청크가 여전히 `min_tokens` 미만이면 이전 청크와
   합치는 로직 추가.
2. `merge_small()`이 원래 `chunk_spans()`에서 **전체 문서의 그룹을 다 모은 뒤 한 번에** 호출돼서,
   `A섹션의 작은 마지막 청크가 B섹션(다른 헤더 밑)의 첫 청크와 합쳐지는` 문제가 있었음(헤더
   경계를 넘어 서로 다른 절의 내용이 한 청크로 섞임) → `chunk_spans()`가 **섹션마다** `merge_small()`을
   개별 호출하도록 수정. SRT는 원래 섹션이 문서 전체 하나뿐이라 동작 변화 없음.

**의도된 트레이드오프(버그 아님, 알아두기)**: 표 하나가 그 자체로 `max_tokens`를 넘는 초대형
표라면, "표는 절대 안 쪼갠다"는 원칙이 크기 제한보다 우선이라 **그 표만 담은 청크가 `max_tokens`를
초과한 채로 나올 수 있음**.

**사용법** (첨부 `smart_chunker.py`, 약 150줄)

```python
from transformers import AutoTokenizer
from smart_chunker import SmartChunker, SmartTextSplitter

# docx_track에서는 임베딩 모델을 또 로드하지 않고, retrieval_benchmark.py가 이미
# 들고 있는 HuggingFaceEmbeddings 인스턴스의 내부 SentenceTransformer(`._client`)를
# 그대로 재사용한다 - GPU에 bge-m3가 두 번 안 올라가게(OOM 예방).
embed = embeddings._client                                   # HuggingFaceEmbeddings 인스턴스에서 추출
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")          # SRT는 Gemma 토크나이저
count = lambda s: len(tok.encode(s, add_special_tokens=False))

smart = SmartTextSplitter(SmartChunker(embed, count, mode="docx", lang="ko",
                                       min_tokens=100, max_tokens=400))
chunks = smart.split_documents(docs)     # 기존 fixed/semantic과 똑같이 사용
```

`mode`는 `"docx"`, `"pdf"`, `"longbench"`, `"srt"` 네 가지입니다. 이게 곧 **문서 유형 라우터**입니다
(데이터셋마다 "무엇을 구조로 볼지"만 다르고 나머지 알고리즘은 동일).

출력은 항상 원문을 그대로 자른 조각(`text[start:end]`)입니다. **이게 실제로 의미 있는 트랙은
SRT/pdf뿐**입니다 - 이 둘의 로더는 지금도 `_Builder`/`metadata["units"]`(오프셋 기반 구조,
직접 확인함)를 쓰기 때문에 `add_start_index`/`fragments()`/`merge_to_units()`가 그대로
동작해야 함. **docx_track은 이번 세션에 이 오프셋 체계를 아예 없앴기 때문에(`doc_name`+내용
F1로만 채점, offset 불필요) 이 성질이 docx에선 그냥 안 쓰이는 부가 기능일 뿐 - 필요해서가
아니라 공짜로 딸려오는 것**.

---

## 3. SRT 트랙: 번역 → 타임스탬프 매핑

**지금 레포 코드가 이미 원하는 동작을 합니다. 그대로 유지하세요.**

```
fixed가 큐 중간을 자른 경우:
  큐 17 = "I was thinking that we should go"
  청크 A 끝: "...I was thinking"      → [5] I was thinking        → 번역 → "내 생각엔"
  청크 B 시작: "that we should go..." → [1] that we should go     → 번역 → "우리가 가야 할 것 같아"
  merge_to_units(): 큐 17 = "내 생각엔 우리가 가야 할 것 같아"  → 원래 타임스탬프로 저장
```

- `srt/mapper.py`의 `fragments()`가 청크와 큐의 겹치는 부분을 찾고,
  `merge_to_units()`가 같은 큐 조각들을 다시 이어 붙이고, `write_srt()`가 원본 타임스탬프로 씁니다.
- 즉 **각 청커가 자른 그대로 LLM에 넣고 결과를 보여줍니다.** fixed가 큐를 자르면 번역이 어색해지는
  것 자체가 실험 결과입니다.
- smart는 큐 경계에서만 자르므로 잘린 큐가 0개입니다.

**추가할 것 (작게)**

1. `subtitle_pipeline.py`의 `default_subtitle_chunkers()`에 `smart` 한 줄 추가
   ```python
   "smart": SmartTextSplitter(SmartChunker(embed_model, gemma_count, mode="srt",
                                           lang=src_lang, min_tokens=128, max_tokens=512)),
   ```
2. 결과표에 두 열 추가: **청크 수**, **잘린 큐 수** (`fragments()`의 4번째 값 `w`가 False인 큐 개수)
3. fixed 500자는 한국어와 영어에서 정보량이 크게 다릅니다. 보고서에 방향별 평균 토큰 수를 같이 적으세요.

---

## 4. docx 트랙 (Allganize): 원본 벤치마크 방식 — **다음에 만질 곳**

Allganize 리더보드 방식은 **"문서 검색 → LLM 답변 → 정답과 비교해 O/X"** 입니다.
원본은 LLM 4개가 투표로 O/X를 정했습니다. 우리는 Gemma 1개로 판정합니다.

이미 `fixed`(chunk_size 256/500/1000)와 `semantic`(percentile 80/90/95) 그리드서치를
211개 QA 전체로 돌려서 최적값을 찾아둔 상태 (`fixed_500`: hit_rate 0.948/MRR 0.886 최고,
`fixed_256`: F1 0.209 최고, `semantic_80`: percentile 중 최고지만 fixed에는 전부 못 미침).
**smart는 이 최적값들과 3파전으로 비교하는 게 다음 단계.**

**평가 순서**

1. 45개 docx → 3개 청커로 청킹 (기존 `retrieval_benchmark.py` 구조 그대로, smart는
   `embeddings._client` 재사용 - 위 2절 참고)
2. 질문으로 top-k 검색 (bge-m3, k=5)
3. **검색 지표** (LLM 없음, 빠름) — `doc_hit@5`, `MRR`: 기존 코드 그대로 (정답 파일에서 나온 청크가 있는가)
4. **답변 지표**: 검색된 청크를 Gemma에 넣어 답변 → Gemma에게 "정답과 같은 의미인가? O/X" 판정
5. `context_type`(paragraph / table / image)별로 따로 표를 만듭니다.
   **스마트 청커의 차이는 table에서 가장 크게 날 것**입니다. image 질문은 그림을 빼고 변환했으므로 참고용입니다.

> **`page_hit@5`는 지금 뺀다 (보류, 원 계획에서 삭제).** 원본 PDF 페이지 텍스트가 필요한데
> `data/allganize/`엔 `docx/`만 있고 **원본 PDF가 로컬에 없음**(실측 확인). `documents.csv`의
> `url`로 새로 받아야 하는데, 그러면 예전에 "9개 법률 문서 파일명 안 맞음"으로 겪었던
> 매칭 문제를 또 거칠 가능성이 있음 - 원본 PDF를 실제로 확보하기 전까지는 선택 항목으로 둔다.

```python
# eval_common.py - 겹침 판정 (페이지·섹션 적중에 공통으로 사용)
def overlap_ratio(chunk_text, gold_text):
    """청크 단어 중 정답 구간에도 있는 단어 비율"""
    c, g = chunk_text.split(), set(gold_text.split())
    return sum(w in g for w in c) / max(len(c), 1)

def is_hit(chunk_text, gold_text, threshold=0.5):
    return overlap_ratio(chunk_text, gold_text) >= threshold
```

> LLM 판정은 틀릴 수 있습니다. 50개 정도는 직접 채점해서 Gemma 판정과 몇 % 일치하는지
> 보고서에 적으면 신뢰도가 올라갑니다. (Allganize도 사람 채점과 약 8% 차이가 있었다고 밝힘)

---

## 5. pdf 트랙 (Vectara Open RAG Bench) — **로더 보강이 먼저 필요**

데이터에 `queries.json`(질문), `qrels.json`(정답 논문 id + **섹션 번호**), `answers.json`,
`corpus/`(논문별 섹션 텍스트), 논문 PDF 50편이 로컬에 실제로 있음(실측 확인, 실행 자체는 바로 가능
- 원 계획의 "100~200편"보다 적으니 아래 1번은 "50편 전부 사용"으로 수정).

**smart를 붙이기 전에 `pdf_track/loader.py`에 먼저 손볼 것 2가지**:

1. **헤더 감지 추가** — 지금 로더(`load_pdf`)는 폰트 크기로 각주만 구분하고(`avg_size <
   footnote_threshold`) 제목은 아예 구분 안 함(`{"heading": False}`가 하드코딩, 실제로
   `True`로 세팅하는 코드가 없음). 헤더 텍스트가 본문 문단에 그냥 섞여 들어가서, 지금
   상태로 smart를 돌리면 `PDF_HEADING_RE`가 찾을 헤더 자체가 텍스트에 안 남아있을 확률이
   큼(구조 1단계가 사실상 무력화). `_dominant_body_size`보다 확연히 큰 폰트 줄을
   `kind="heading"`으로 별도 flush하도록 추가해야 함.
2. **표를 JSON 대신 마크다운으로** — 지금은 표 행마다 `json.dumps({...})`를 `kind="row"`
   Unit으로 저장. `smart_chunker.py`의 표 감지(`body.startswith(("|", "<table", "{"))`)는
   `|`도 이미 인식하므로, PyMuPDF가 주는 `rows`/`header`로 마크다운 표 문자열을 만들어
   `kind="table"` Unit 하나로 저장하면 (a) docx와 표현 방식이 통일되고 (b) bge-m3 임베딩
   입장에서 JSON보다 자연스러운 텍스트라 의미 경계 판단에 유리하고 (c) `smart_chunker.py`는
   손댈 필요가 없음(`|`-prefix 병합 로직이 이미 있음). 이 JSON 포맷을 읽는 다운스트림
   코드는 현재 전혀 없음(실측 확인, `pdf_track/mapper.py`는 kind에 무관하게 동작) - 안전한 변경.

**평가 순서** (위 로더 보강 후)

1. 로컬에 있는 논문 PDF 50편 전부 사용 + 해당 질문들
2. 보강된 `pdf_track/loader.py`로 PDF 로드 → 3개 청커로 청킹
3. 질문으로 top-5 검색 → 정답 섹션 텍스트(`corpus/`에서 가져옴)와 `is_hit()`로 비교
4. 지표: `section_hit@5`, `MRR`, (여유 있으면) 답변 생성 후 O/X
5. 질문 유형(text-only / text-table)별로 나눠 보고. 이미지 질문은 제외

주의: `corpus/`의 섹션 텍스트는 **채점에만** 쓰고 청커에는 넣지 않습니다(정답을 미리 보는 셈이라).

> `pdf_track/mapper.py`를 손대는 김에: 지금 `chunk_bundle()`이 문서마다 `split_documents([doc])`를
> 따로 호출하는 per-doc 루프인데, 이건 docx_track에서 이미 "`split_documents()`에 여러 문서를
> 한번에 넘겨도 문서 경계를 안 넘는다"는 걸 실측으로 증명하고 지운 것과 똑같은 불필요한
> 패턴 - 같이 정리하면 일관성 있음(이번 계획의 핵심은 아니라 선택 사항).

---

## 6. LongBench 트랙: 공식 방식 그대로 — 순서상 제일 마지막

> LongBench는 "삭제"된 게 아니라, 2026-09-22 세션에 "폴더만 만들어두고 구현은 나중에"로
> **의도적으로 미뤄둔 상태**였음(`src/longbench/`엔 `__init__.py`만 있음, 실측 확인). 이 계획이
> 이걸 다시 꺼내는 건 번복이 아니라 원래 예정된 순서를 재개하는 것 - 다만 4개 트랙 중
> 가장 나중(SRT/docx/pdf 끝난 뒤)에 시작.

LongBench 논문에도 **"긴 문맥을 청크로 잘라 검색해서 모델에 넣는"** 실험이 있습니다
(200단어 청크 top-7 / 500단어 청크 top-3). 이 설정을 그대로 따라 하면 됩니다.

| 하위셋 | 유형 | 공식 지표 |
|---|---|---|
| qasper, multifieldqa_en | 단일 문서 QA | QA F1 |
| hotpotqa, 2wikimqa | 여러 문서 QA | QA F1 |
| narrativeqa | 소설 QA | QA F1 |
| (선택) gov_report | 요약 | ROUGE-L |

1. `datasets.load_dataset("THUDM/LongBench", name)`으로 로드. 하위셋당 100개
2. 샘플마다 `context`를 3개 청커로 청킹 → `input`(질문)으로 검색
3. **검색된 청크를 약 1,500단어(≈2,000토큰)까지** 담아 Gemma에 넣음 (모든 청커 동일)
4. 프롬프트는 LongBench 공식 `dataset2prompt.json`, 채점은 공식 `metrics.py`의
   `qa_f1_score`를 복사해서 사용 → 다른 논문 숫자와 비교 가능
5. (선택) gov_report는 "청크마다 요약 → 요약들을 다시 요약"(map-reduce)으로 RFP의
   "요약 품질" 비교에 사용

---

## 7. 공정한 비교를 위한 규칙 (보고서에 그대로 적기)

1. 임베딩(bge-m3), 검색(top-k 또는 같은 토큰 예산), LLM(Gemma, greedy), 프롬프트를 모두 고정
2. 청커 크기는 **평균 청크 토큰 수가 비슷하게** 맞춤 (예: 모두 300~400토큰 근처)
3. 모든 결과표에 **청크 개수 / 평균 토큰 수**를 같이 적음 (큰 청크가 유리해 보이는 착시 방지)
4. 파라미터(max_tokens, percentile 등)는 데이터 일부(20%)로만 고르고, 나머지로 최종 점수 계산

---

## 8. 폴더 구조 (지금 구조에서 최소 변경)

> **`chunkers/baselines.py`(fixed까지 래퍼로 감싸는 공유 폴더) 안 씀.** 이번 세션에
> docx_track의 `splitters.py`(`make_fixed_splitter` 같은 순수 pass-through 래퍼)를 "의미
> 없는 간접화"라고 판단해서 지우고 `CharacterTextSplitter`/`SemanticChunker`를 트랙마다
> 직접 호출하게 바꿨음 - 공유 폴더를 새로 만들면 그 결정을 되돌리는 셈. **진짜 커스텀
> 로직(`smart_chunker.py`, srt의 `SemanticTextSplitter`)만 공유하고, fixed/semantic은
> 지금처럼 각 트랙이 LangChain 클래스를 직접 호출한다.**

```
LLM_chunking/
├─ requirements.txt          # ★ langchain-core, langchain-text-splitters, langchain-experimental,
│                            #   langchain-huggingface, docling, pymupdf, datasets 추가
├─ src/
│  ├─ config.py, common.py, llm.py, indexing.py   # 그대로
│  ├─ smart_chunker.py        # 새로: 첨부 파일 (공유 - mode 파라미터로 4트랙 라우팅)
│  ├─ eval_common.py          # 새로: overlap_ratio, is_hit, MRR, 토큰 예산 검색
│  ├─ srt/                    # 그대로(splitters.py의 SemanticTextSplitter 포함) +
│  │                          # subtitle_pipeline.py에 smart 한 줄 추가
│  ├─ docx_track/
│  │  ├─ loader.py            # 그대로
│  │  └─ retrieval_benchmark.py  # fixed/semantic/smart 3파전 (page_hit은 보류)
│  ├─ pdf_track/
│  │  ├─ loader.py            # 헤더 감지 + 마크다운 표로 보강 (5절)
│  │  └─ vectara_benchmark.py # 새로
│  └─ longbench/
│     ├─ benchmark.py         # 새로: 로드 → 청킹 → 검색 → Gemma → F1
│     └─ metrics.py           # LongBench 공식 metrics.py 복사
├─ tests/
│  └─ test_srt_roundtrip.py   # "번역" 대신 원문 그대로 넣어서 원본 SRT가 복원되는지 확인
└─ results/
```

---

## 9. 일정 (중간발표 8주차 기준)

| 주차 | 할 일 | 결과물 |
|---|---|---|
| 5 | `smart_chunker.py` 넣기, requirements 정리. **docx_track부터**: fixed/semantic 최적값(이미 그리드서치로 확보) + smart 3파전 | Allganize 검색 결과표 (fixed/semantic/smart) |
| 6 | pdf_track 로더 보강(헤더 감지 + 마크다운 표) → 3개 청커 × 검색 지표(section_hit, MRR) | Vectara 검색 결과표 |
| 7 | SRT에 smart 한 줄 추가해서 5편 돌리기 + LongBench 구현 시작(QA 2개 하위셋) | 4개 데이터셋 1차 표 |
| 8 | **중간 발표**: 어디서 이기고 어디서 지는지 | 중간 슬라이드 |
| 9 | 답변 생성 평가 (Allganize O/X, LongBench F1 전체) | 2차 결과표 |
| 10 | 어블레이션: 스마트 청커에서 ①구조 끄기 ②창 크기 1 vs 2 ③min/max 바꾸기 | 어블레이션 표 |
| 11 | 엣지 케이스: 깨진 타임스탬프, cp949, 빈 큐, 아주 큰 표 | 테스트 코드 |
| 12–13 | 코드 정리, README 재현 절차, 결과 재현 확인 | GitHub 정리 |
| 14 | 데모 + 최종 발표 | Tech Report |

**어블레이션이 곧 "내 알고리즘의 어느 부분이 효과가 있었나"의 증거**입니다.
스마트 청커가 3단계라서 단계별로 하나씩 끄면 표 하나가 나옵니다.

---

## 10. 예상되는 문제와 대처 (보고서의 트러블슈팅 절에 쓸 것)

| 문제 | 원인 | 대처 |
|---|---|---|
| `semantic_90/95`에서 OOM | percentile이 높으면 경계가 적어져 청크가 매우 커지고, bge-m3가 긴 청크를 한 배치에 여러 개 임베딩 | `encode(batch_size=4)`, 임베딩 전에 길이순 정렬, fp16 |
| Gemma와 bge-m3 동시 로드 시 메모리 부족 | 24GB 한 장 | 검색 결과를 파일로 저장 → 임베딩 모델 내리고 Gemma 로드 |
| 번역 출력에 `[n]` 번호 누락 | 청크가 너무 길거나 모델이 합쳐서 번역 | 누락 개수를 결과표에 기록, 청크 크기 줄이기 |
| 스마트 청커가 이기지 못하는 데이터셋 | 서술형 텍스트(narrativeqa)는 구조가 없음 | 정상적인 결과. "구조가 있는 문서에서만 효과"라고 솔직히 적기 |

> **실제로 이 표의 첫 줄을 그대로 겪음(2026-09-24, docx_track 그리드서치 중).**
> `semantic_90`에서 정확히 예측된 지점에 CUDA OOM 발생 — 다만 실측해보니 우리 자신의
> 배치 크기보다도 **공유 GPU에 동시에 떠 있던 외부 프로세스(16.57GiB 사용 중)와의 경합이
> 더 직접적인 원인**으로 보였음(`nvidia-smi`로 그 프로세스가 크래시 직후 매번 사라진 걸
> 확인, 재현 시도마다 다른 PID인데 메모리 사용량은 동일 - 이 서버에 주기적으로 도는 다른
> 작업으로 추정). 적용한 완화책은 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(에러
> 메시지가 직접 권장)였고 이후 재현 안 됨. **이 표가 제안한 `encode(batch_size=4)` +
> 길이순 정렬도 우리 쪽 최대 배치 크기 자체를 줄여서 외부 경합에 더 안전해지므로 같이
> 적용할 가치 있음** — 두 대처가 상호 배타적이지 않고 상호 보완적.

---

## 참고 자료

- [KT Cloud Tech Blog — RAG 청킹 전략과 최적화](https://tech.ktcloud.com/entry/2025-11-ktcloud-rag-ai-%EC%B2%AD%ED%82%B9%EC%A0%84%EB%9E%B5-%EC%B5%9C%EC%A0%81%ED%99%94): 고정/의미/구조 기반 비교, Recall@k·nDCG 지표
- [forge-tutorial-rag 4.5 청킹 전략 비교 실험](https://github.com/jsonpassion/forge-tutorial-rag): Recursive(400, overlap 0) 베이스라인 우선, 구조+재귀 2단계 분할
- [Is Semantic Chunking Worth the Computational Cost? (Vectara, NAACL 2025)](https://arxiv.org/abs/2410.13070)
- [Chroma — Evaluating Chunking Strategies for Retrieval](https://www.trychroma.com/research/evaluating-chunking)
- [Meta-Chunking (arXiv 2410.12788)](https://arxiv.org/abs/2410.12788), [MoC (ACL 2025)](https://arxiv.org/abs/2503.09600): LLM 기반 청킹 관련 연구 (관련 연구 절에 인용)
- [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench)
- [LongBench (THUDM)](https://github.com/THUDM/LongBench)
