# 스마트 청킹 · 번역 · 병합 파이프라인 (프로젝트 2-1) — 최종 설계

> 기준 레포: `ddxx-dd/LLM_chunking` · 최초 작성 2026-09-24, 전면 재작성 2026-09-25(2차)
> 이전 버전(Docling+python-docx 분리안, Phase1/Phase2안)을 모두 대체한다.

## 0. 목표와 원칙

**핵심 질문**: fixed(글자수 분할) / semantic(의미 분할) / smart(구조+의미+크기 분할) 세 청커가
검색·요약·번역 결과에 어떤 영향을 미치는가?

**원칙**:
1. 네 데이터셋(SRT, Allganize-docx, Vectara-pdf, LongBench)에서 세 청커를 같은 조건으로 비교
2. **번역 채점은 항상 요소(표는 셀) 단위** — 청크를 통째로 채점하지 않는다. fixed/semantic도
   예외 아님(이번에 확정됨 — 처음엔 smart만이라고 했었는데, 공정한 비교를 위해 세 청커 다 같은
   단위로 채점해야 한다는 게 맞음)
3. 번역 결과를 원본 구조 그대로 파일에 되돌려 넣는 "병합"은 **smart만** 수행
4. 문서 하나당 파싱은 1번만, JSON 캐싱
5. 파싱 직후 **항등 테스트**(원문 그대로 되돌려 썼을 때 원본과 같은지) 통과 전에는 다음 단계로
   안 넘어감
6. 학부 3학년 프로젝트 규모 — 서식 보존/번역 재시도/폰트 서브셋은 스트레치 골

---

## 1. 데이터셋 현황

| 트랙 | 데이터 | 상태 |
|---|---|---|
| SRT | OpenSubtitles 5편 | **완료**, 이 설계와 무관 |
| docx | Allganize, 45개 문서·211 QA | 검색용 그리드서치 완료, **파서 재작성 필요**(아래 3절) |
| pdf | Vectara, 50개 논문·273 QA | 로더는 Docling으로 교체됨, **파서 재작성 필요**(아래 4절) |
| LongBench | THUDM/LongBench | 미구현, 이 설계와 무관 |

---

## 2. 문서 객체 모델 (`docobj.py`)

```python
elements = [
    {"id": 0, "label": "section_header", "level": 1, "text": "...", "loc": {...}, "span": (0, 23)},
    {"id": 1, "label": "text", "level": None, "text": "...", "loc": {...}, "span": (24, 110)},
    {"id": 2, "label": "table", "text": "<마크다운 표 전체>", "loc": None, "span": (111, 400),
     "table_id": 0, "caption_ids": [3], "n_rows": 4, "n_cols": 3,
     "cells": [
        {"cell_id": 0, "row": 0, "col": 0, "row_span": 1, "col_span": 1, "header": True,
         "text": "구분", "span_in_table": (2, 4), "loc": {...}},
        ...
     ]},
]
```

**두 가지가 새로 추가됨**:
1. **`span`**: 모든 요소가 "이어붙인 평문(`flat_text`)" 안에서 차지하는 (시작,끝) 문자 오프셋.
   `flat_text = "\n".join(e["text"] for e in elements)`로 만들면서 기록 — fixed/semantic
   청커가 이 평문을 자르면, 청크의 `start_index`(LangChain `add_start_index=True`로 얻음)와
   `span`을 비교해서 "이 청크가 어느 요소(들)를 걸쳤는지"를 알 수 있다. **이게 있어야
   fixed/semantic도 번역을 요소 단위로 채점할 수 있음**(1절 원칙 2번의 근거).
2. **표는 이제 "표 요소 1개 + `cells` 목록"의 두 층 구조**(6절 참고) — 셀을 낱개 요소로
   펼치지 않는다. 표 하나 = elements 리스트에서 `label:"table"`인 요소 딱 1개.

`loc`(요소 위치, 병합 쓰기용): docx는 `{"paraId": "..."}`, pdf는 `{"page":, "bbox":{...}}`.

---

## 3. docx 파싱 (`docx_track/parse.py`)

### 순서

```
1. python-docx로 원본 열기 → 모든 문단에 w14:paraId 없으면 부여 → *.anchored.docx로 저장
   (python-docx의 nsmap에 w14가 이미 등록되어 있음 - 확인됨)

2. MsWordDocumentBackend(in_doc, anchored_path).convert() 직접 호출
   → backend.paragraph_to_items: dict[etree._Element, list[RefItem]] 를
     "self_ref → paraId" 역인덱스로만 사용한다 - 이걸 그대로 순회해서 elements를 만들면
     안 됨(표가 본문 흐름과 다른 시점에 paragraph_to_items에 채워질 수 있어서, 그대로
     쓰면 표가 문서 끝으로 밀려나 smart의 제목 경계·캡션 묶기가 깨짐).

3. **진짜 순회는 doc.iterate_items()로** - 이게 진짜 문서 순서(제목→본문→표→본문...)를 보존함.
   각 아이템의 self_ref로 역인덱스에서 paraId를 찾는다.
   RefItem은 cref만 갖고 있으므로(RefItem.model_fields == ['cref'], 실측 확인) doc.resolve()로
   실제 아이템을 얻어야 함.

4. 같은 paraId에서 여러 RefItem이 나오는 경우(문단 하나가 서식 등으로 여러 Docling 텍스트
   조각으로 쪼개진 "인라인 그룹") → 문단 하나 = 요소 하나로 합침(텍스트 이어붙임).

5. 조상에 w:tbl이 있는 문단(표 셀 안의 문단)은 별도 "text" 요소로 만들지 않음 -
   표 요소의 cells 안에서만 다뤄짐(중복 방지).

6. 텍스트 상자 안 문단도 body.iter(w:p)로 찾아지므로 paraId가 정상적으로 붙는다 -
   본문과 똑같이 병합 가능(다만 이번 세션에 직접 실행 검증은 안 했음, 설계 단계).
```

### skip 규칙 (번역·병합만 제외, 청킹·검색·요약엔 포함)
- 공통: `formula`, `picture`, `page_header`/`page_footer`(furniture)
- docx 전용: 문단에 `w:fldChar`/`w:instrText`(필드 코드) / `w:footnoteReference` / `w:ins`·`w:del`(변경추적)

### 헤더 감지
스타일 기반(`Heading N`/`Title`). 45개 중 43개(95.6%)가 실제 스타일 보유(실측). 예외 2개는
알려진 한계.

---

## 4. pdf 파싱 (`pdf_track/parse.py`)

```python
result = _converter.convert(filepath)   # do_ocr=False, TableFormerMode.FAST
doc = result.document
for item in doc.texts:
    provs = []
    for p in item.prov:                      # ★ prov 전체를 저장(문단이 페이지/단을 넘으면 여러 개)
        page_h = doc.pages[p.page_no].size.height
        bbox = p.bbox if p.bbox.coord_origin.name == "TOPLEFT" else p.bbox.to_top_left_origin(page_h)
        provs.append({"page": p.page_no, "bbox": {...}, "charspan": p.charspan})
    elements.append({..., "loc": {"prov": provs}})   # loc이 리스트를 담을 수 있음
```
**이전 버전과 차이**: `item.prov[0]`만 쓰던 걸 **`item.prov` 전체**로 바꿈 — 문단이 페이지나
단(2단 논문)을 넘으면 prov가 여러 개 나오기 때문.

### 표: 두 층 구조 (6절과 동일 원칙)
`doc.tables`의 표 하나 = elements의 `label:"table"` 요소 1개. `table.data.table_cells`를
순회해서 `cells` 리스트를 만들고, 표 전체 텍스트는 마크다운으로 직렬화(6절 참고).

### 수식 글꼴 skip (pdf 전용, 참고 패턴 — 검증 필요)
```
(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Sym|.*Math)
```
`.*Ital`/`.*Mono`/`.*Code`는 **제외하고 시작** — `.*Ital`은 일반 이탤릭 본문(Times-Italic 등)까지
잡아버림. `page.get_text("dict", clip=bbox)`로 글꼴명을 읽어 판정. **이 패턴은 PDFMathTranslate의
`converter.py`에서 유래했다고 하는데, 이번 세션에 그 소스를 직접 열어본 적은 없음(미검증)** —
우리 50개 코퍼스 중 수식이 있는 논문 2~3개로 실제 겹치는지 검증 후 확정할 것.

### 실측 결과 (변경 없음)
50개 전체 스캔: 실패 0, `section_header` 0개 문서 0, 요소 수 30개 미만 문서 0. **페이지 넘는
표**: `table.prov` 길이 1 초과(Docling 자체 인식) 0건. 대신 "인접 페이지 열 개수 일치"로 찾은
분할 의심 59건 중 `2404.09358v3.pdf`(7열, 11페이지 연속), `2410.22706v2.pdf`(6~9열, 여러
페이지 연속)는 실제 분할된 표일 가능성이 높음(우연이라기엔 규칙적, 두 문서 다 `table_cell`
수가 전체 중 최다 수준이었음) — smart `group_elements()` 구현 후 이 두 문서를 스팟체크할 것.

---

## 5. 청킹 (`smart_chunker.py`)

### fixed/semantic — 이제 요소 단위 채점을 위해 오프셋을 추적함

```python
flat_text = "\n".join(e["text"] for e in elements if e["label"] not in FURNITURE)
# elements에 span 기록(위 2절)

fixed_chunks = CharacterTextSplitter(separator="", chunk_size=500,
                                      add_start_index=True).create_documents([flat_text])
# ★ 번역용 semantic은 LangChain SemanticChunker를 쓰지 않는다 - " ".join() 재조합으로
#   오프셋이 깨지는 게 이미 이 레포에서 실측 확인된 문제. 대신 레포의 오프셋 보존판
#   SemanticTextSplitter(srt/splitters.py, add_start_index=True 기본)를 그대로 재사용.
#   (주의: docx_track 검색 전용 그리드서치는 여전히 LangChain SemanticChunker를 씀 -
#   검색은 오프셋이 필요 없어서 그대로 두고, "번역"에서만 오프셋 보존판으로 바꾸는 것)
semantic_chunks = SemanticTextSplitter(embed_model, ...).create_documents([flat_text])

for chunk in fixed_chunks:  # semantic_chunks도 동일
    start, end = chunk.metadata["start_index"], chunk.metadata["start_index"] + len(chunk.page_content)
    covered = [e for e in elements if e["span"][0] < end and start < e["span"][1]]
    # covered의 각 요소(또는 겹치는 부분)에 [n] 번호 부여해서 번역 → 요소 단위로 CometKiwi 채점
```

### smart — `group_elements()`
```python
def group_elements(elements, count_tokens, min_tokens=100, max_tokens=400, window=2):
    # 구조 우선: section_header 경계에서 새 그룹, 표+캡션(caption_ids로 연결)은 한 그룹
    # 표 하나가 max_tokens를 넘으면: 행 묶음으로 나누고, 나뉜 조각마다 헤더 행을 반복해서 앞에 붙임
    # 의미 분할: 그 외 그룹이 max_tokens 넘으면 요소 경계(중간 아님)에서 좌우 window 유사도 최저점 분할
    # 크기 조절: min_tokens 미만 그룹은 이웃과 병합
    return [{"element_ids": [...], "text": "..."}]
```

### 세 청커 공정성 (중요)
- 세 청커 다 **정확히 같은 elements 집합, 같은 `flat_text`**에서 출발(furniture 제외는 공통,
  formula/필드코드 등은 청킹엔 포함).
- 표의 `text`(마크다운 직렬화)는 파싱 시점에 한 번 정해지고, 세 청커 다 같은 표현을 씀 —
  fixed/semantic은 표 중간에서 잘릴 수 있음(이게 비교 포인트, 의도된 것). smart만 표를 안 자름.

---

## 6. 표: "표 요소 1개 + 셀 목록" 두 층 구조

셀을 낱개 요소로 펼치면 셀 간 관계·헤더·병합·캡션 연결이 사라진다(지난번 발견한 문제).
그래서 표는 항상 이렇게 저장:

```python
{"id":.., "label": "table", "table_id": 0, "caption_ids": [..],
 "n_rows":.., "n_cols":.., "text": "<마크다운 표>",   # 세 청커 공통 직렬화
 "cells": [
    {"cell_id":.., "row":.., "col":.., "row_span":.., "col_span":.., "header": bool,
     "text": "..", "span_in_table": (start,end),  # 이 셀 글자가 표 text 문자열 안에서 차지하는 위치
     "loc": {...}},                                # pdf: 셀 bbox / docx: 셀 첫 문단 paraId
    ...
 ]}
```
병합 셀은 한 번만 기록(Docling `TableCell`의 시작 행·열 + span 사용, docx도 python-docx가
이미 병합을 해석해서 주므로 동일 원칙).

**청킹·검색·요약**: 표 요소의 `text`(마크다운) 사용. **번역·채점·병합만 셀 단위**: 번역 청크가
걸친 셀(`span_in_table` 기준)마다 `[n]` 번호 부여, 숫자/기호만 있는 셀은 skip(원문 유지).
선택 사항으로 표 전체를 "참고용, 번역 금지" 문맥으로 프롬프트 앞에 붙여 셀 번역에 맥락 제공
가능. 병합(smart만)은 각 셀의 `loc`에 씀.

---

## 7. 병합 (`docx_track/writer.py`, `pdf_track/writer.py` — smart 번역만)

### docx
```python
doc = DocxDocument(anchored_path)   # ★ 원본이 아니라 *.anchored.docx를 열어야 함(paraId가 거기만 있음)
for eid, translated_text in translations.items():
    para = find_paragraph_by_paraId(doc, elements[eid]["loc"]["paraId"])
    for content in para.iter_inner_content():   # run + 하이퍼링크 모두 순회(p.runs만 쓰면
                                                  # 링크 글자가 안 지워지는 버그가 있었음)
        ...  # 1차: 첫 run에 번역문 전체, 나머지는 비움. cell.merge() 호출 안 함.
doc.save(f"results/docx/{name}_translated.docx")
```

### pdf — 페이지마다 정해진 순서로 처리(bbox 겹침 시 방금 쓴 걸 지우는 사고 방지)
```python
for page_no, page_elements in group_by_page(translations):
    page = pdf[page_no - 1]
    # (a) 빈 임시 페이지에 먼저 넣어보고 들어가는지 확인
    fits = [e for e in page_elements if test_fit(scratch_page, e["bbox"], e["translated_text"])]
    # (b) 들어가는 요소만 전부 redact 표시
    for e in fits:
        page.add_redact_annot(fitz.Rect(e["bbox"]).__add__/*1pt 안쪽*/)
    # (c) 한 번에 적용
    page.apply_redactions(images=PDF_REDACT_IMAGE_NONE, graphics=PDF_REDACT_LINE_ART_NONE)
    # (d) 그다음에 전부 쓰기
    for e in fits:
        spare_height, scale = page.insert_htmlbox(rect, e["translated_text"],
            css="font-family:NotoSansKR", archive=pymupdf.Archive("fonts/"), scale_low=0.6)
        # ★ 반환값은 (spare_height, scale) 튜플. spare_height<0이면 아무것도 안 쓰인 것(실패)
        if spare_height < 0:
            log_failed(e["id"]); continue   # 원문 유지
        log_scale(e["id"], scale)
pdf.subset_fonts()   # 맨 마지막에 한 번
pdf.save(f"results/pdf/{name}_translated.pdf")
```
**여러 칸(prov 여러 개)에 걸친 문단**: 1차 구현은 **첫 번째 칸에만 쓰고 나머지 칸은 원문
유지**(charspan 값이 항상 신뢰 가능한지 미확인 — `document.py:4642`에 `charspan=(0,0)`
placeholder로 쓰는 코드가 있는 걸 봐서 항상 의미있는 값은 아닐 수 있음). charspan 비율로
쪼개 배치하는 건 스트레치 골로 미룸.

---

## 8. 평가

| 대상 | 방법 |
|---|---|
| 검색 | doc_hit@5/section_hit@5, MRR, F1 |
| 요약 | QA 커버리지(Gemma O/X) |
| 번역 | **CometKiwi, 요소(표는 셀) 단위 — 세 청커 전부**(fixed/semantic 포함, 1절 원칙 2번). `.venv-eval`에서 저장된 (원문,번역) 쌍 읽어서 채점. 실제 작동 확인됨(테스트 점수 0.888) |
| 병합 | **항등 테스트가 최우선**(파싱 직후 실행, 통과해야 다음 단계) — docx는 텍스트 완전 일치, pdf는 페이지 텍스트/이미지 거의 일치. 그다음 번역 커버리지, 잘린 요소 수, 번호 누락 수, pdf 축소 배율/실패 수, skip 사유별 수 |
| 공통 | 청크 수, 평균 토큰 수, 청킹 시간 |

CometKiwi는 `.venv311`(번역 실행) → 결과 파일 저장 → 프로세스 종료 → `.venv-eval`(채점)
순서로 완전히 분리해서 실행(같은 프로세스에서 절대 같이 못 씀).

---

## 9. 폴더 구조

```
LLM_chunking/
├─ requirements.txt          # docling==2.127.0 정확히 고정(내부 API 의존)
├─ requirements-eval.txt     # unbabel-comet>=2.0.0
├─ src/
│  ├─ config.py, common.py, llm.py, indexing.py   (그대로)
│  ├─ docobj.py              # elements 스키마(표 2층 구조 포함) + save/load
│  ├─ smart_chunker.py       # group_elements() 추가, 기존 코드 유지
│  ├─ pipeline.py            # --dataset --task 공통 실행 진입점
│  └─ merge_checks.py        # 항등 테스트 등
├─ docx_track/
│  ├─ parse.py               # paraId 앵커링 + iterate_items 순회 + 표 2층
│  ├─ writer.py               # python-docx 병합(anchored 파일 사용)
│  └─ retrieval_benchmark.py / translate_benchmark.py / summary_benchmark.py
├─ pdf_track/
│  ├─ parse.py               # prov 전체 + 표 2층 + 수식 글꼴 skip
│  ├─ writer.py               # PyMuPDF 병합(test-fit→redact→apply→insert 순서)
│  └─ retrieval_benchmark.py / translate_benchmark.py / summary_benchmark.py
├─ evals/comet_score.py      # .venv-eval 전용
├─ fonts/                    # NotoSansKR 등
├─ srt/, longbench/          (그대로/미구현, 무관)
├─ .venv311/, .venv-eval/    (둘 다 이미 존재)
└─ data/processed/           # <문서명>.json 캐시
```

---

## 10. 구현 순서 (확정)

| 순서 | 할 일 | 통과 조건 |
|---|---|---|
| 1 | docx parse(paraId 앵커 + iterate_items 순서 + 표 2층) / pdf parse(prov 전체 + 표 2층) | 코드 작성 |
| 2 | **항등 테스트** (docx·pdf) | 원문 그대로 되돌려 쓴 결과가 원본과 일치 — **통과해야 3번으로** |
| 3 | 45개·50개 전체 재스캔 | label 분포, 표 수, skip 수 확인 |
| 4 | `group_elements()` | 표+캡션 안 잘리는지, 큰 표 헤더 반복되는지 육안 확인 |
| 5 | 검색·요약(세 청커) | 기존 지표로 비교 |
| 6 | 번역 `[n]`(세 청커, 요소/셀 단위 채점) | 번호 누락 개수 기록 |
| 7 | smart 병합 + 병합 지표 | 커버리지, 잘린 요소 수, skip 사유별 수, 숫자 보존율 |
| 8 | 스트레치(시간 남으면) | 서식 태그화, 번역 재시도, pdf 여러 칸 문단 나누기, 수식 자리표시자 |

---

## 11. 트러블슈팅 기록 (보고서용)

| 문제 | 원인 | 대처 |
|---|---|---|
| Docling PDF 변환 시 `CUDNN_STATUS_NOT_INITIALIZED` | 드라이버/cuDNN 버전 불일치 | `torch.backends.cudnn.enabled = False` |
| `semantic_90/95`에서 OOM | 공유 GPU 경합(실측) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| CometKiwi가 transformers 5.x와 충돌 | `unbabel-comet`이 `transformers<5.0` 강제 | `.venv-eval` 분리 — 실제 작동 확인됨 |
| markitdown: 단어 사이 공백 소실 | pdfminer 계열 커닝 처리 문제 | 후보 제외 |
| marker-pdf: 환경 자체에서 실행 실패 | GPU모드 Docker 필요, CPU모드 내부 서버 500 | 후보 제외 |
| python-docx 단독으로 docx 읽으면 텍스트 깨짐 | 겹친 텍스트박스+본문이 섞여 읽힘(실측) | Docling 유지 + paraId 앵커링 |
| pdf 표가 페이지 경계를 넘음 | Docling 표 구조 모델이 페이지 단위 독립 처리(소스 확인) | **실제 사례 확인됨**(2404.09358v3.pdf, 2410.22706v2.pdf) — `group_elements()` 구현 후 스팟체크 |

---

## 참고 자료

- [KT Cloud Tech Blog — RAG 청킹 전략과 최적화](https://tech.ktcloud.com/entry/2025-11-ktcloud-rag-ai-%EC%B2%AD%ED%82%B9%EC%A0%84%EB%9E%B5-%EC%B5%9C%EC%A0%81%ED%99%94)
- [forge-tutorial-rag](https://github.com/jsonpassion/forge-tutorial-rag)
- [Is Semantic Chunking Worth the Computational Cost? (Vectara, NAACL 2025)](https://arxiv.org/abs/2410.13070)
- [Chroma — Evaluating Chunking Strategies for Retrieval](https://www.trychroma.com/research/evaluating-chunking)
- [Meta-Chunking (arXiv 2410.12788)](https://arxiv.org/abs/2410.12788), [MoC (ACL 2025)](https://arxiv.org/abs/2503.09600)
- [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench)
- [LongBench (THUDM)](https://github.com/THUDM/LongBench)
- [CometKiwi](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) — 실제 작동 확인됨
- PDF 번역 조립(미검증, 참고): [Artifex PyMuPDF 가이드](https://artifex.com/blog/translating-pdfs-a-practical-pymupdf-guide), [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate)
