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
| docx | Allganize, 45개 문서·211 QA | ~~검색용 그리드서치 완료~~ → **폐기, 재실행 필요**(파서가 새로 바뀌어서 — paraId 앵커링 + 표 2층 구조. semantic 구현 자체는 원래 쓰던 LangChain `SemanticChunker`로 유지됨, 5절 참고). **파서 재작성 필요**(아래 3절) |
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
         "text": "구분", "span_in_table": (2, 4), "span_abs": (113, 115), "loc": {...}},
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
   펼치지 않는다. 표 하나 = elements 리스트에서 `label:"table"`인 요소 딱 1개. 각 셀은
   **위치를 두 가지로 저장**: `span_in_table`(표 자체의 마크다운 `text` 문자열 안에서의
   위치)과 `span_abs`(`flat_text` 전체 기준 절대 위치, `table.span[0] + span_in_table`로
   계산). 전자는 표 청크 안에서 셀을 찾을 때, 후자는 fixed/semantic 청크가 표 중간을
   가로질렀을 때 그 청크가 어느 셀까지 걸쳤는지 판정할 때 쓴다.

`loc`(요소 위치, 병합 쓰기용): docx는 `{"paraId": "..."}`, pdf는 `{"prov": [{"page":,"bbox":{...}}]}`.

**furniture(`page_header`/`page_footer`)는 파싱 단계에서 elements에 아예 넣지 않는다**(docx도
pdf처럼 동일하게). `flat_text`는 이렇게 필터링된 elements로만 만들어지므로 span 계산 기준이
하나로 고정된다. **skip 대상(formula/picture/필드코드 등)은 furniture와 다르다** — elements와
`flat_text`엔 그대로 남고, 청킹·검색·요약엔 쓰이되 번역·병합 단계에서만 제외된다.

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

   **★ 실측 확인(2026-09-26) — 위 역인덱스만으로는 부족함이 드러남**: `paragraph_to_items`는
   Docling 댓글-연결 기능의 부산물이라 "서식 안 섞인 단일 run 문단"만 커버 -
   `_handle_text_elements()`의 list_item 분기가 항상 빈 elem_ref로 일찍 return해서(소스
   확인) list_item은 등록 자체가 안 되고, 서식 혼합 문단도 마찬가지. 이 상태로 파싱하면
   샘플 문서 211개 중 72개(34%)가 loc을 못 찾음. **해결**: `MsWordDocumentBackend`를
   상속한 `AnchoredWordBackend`로 `_handle_text_elements` 자체를 감싸서, 호출 전후
   `doc.texts` 길이를 비교해 그 호출에서 새로 생긴 텍스트 객체마다
   `obj_to_paraid[id(객체)] = 그 문단의 paraId`를 직접 기록 - list_item/서식 혼합이든
   내부적으로 이 메서드 한 번의 호출 안에서 `doc.texts`에 append되므로 100% 잡힌다.
   self_ref가 아니라 객체 id()를 키로 쓰는 이유: convert() 뒷단에서 furniture 삭제 등으로
   self_ref 문자열이 재계산될 수 있어 같은 파이썬 객체(id)를 키로 써야 안전함. 이 방법으로
   72개 → 20개(전부 picture, 즉 원래도 loc이 없는 게 정상)로 줄어 **실질적으로 0건**이
   됨(3절 아래 "표 셀의 loc" 앞부분 참고).

4. 같은 paraId에서 여러 RefItem이 나오는 경우(문단 하나가 서식 등으로 여러 Docling 텍스트
   조각으로 쪼개진 "인라인 그룹") → 문단 하나 = 요소 하나로 합침(텍스트 이어붙임).

5. **★ 실측 확인(2026-09-26, 항등 테스트에서 발견) — 애초 가정이 틀렸음이 드러남**:
   "`doc.iterate_items()`가 표 셀 안 문단은 표와 별개로 top-level 아이템을 안 만든다"는
   가정으로 명시적 필터를 안 넣었었는데, 실제로는 표 셀 문단이 표 요소와 **별개로 일반
   "text" 아이템으로도 다시 나온다**(실측: 표 하나의 셀 6개가 전부 중복 생성됨 → 같은
   paraId를 가진 요소가 2개씩 생겨서 병합 시 나중 것이 먼저 것을 덮어씀). **해결**: anchored
   문서의 모든 `<w:p>` 중 조상에 `w:tbl`이 있는 것들의 paraId 집합(`table_paraids`)을
   미리 만들어두고, 일반 텍스트 루프에서 `para_id in table_paraids`면 명시적으로
   건너뛴다 - 표 셀은 오직 표 요소의 `cells` 안에서만 다뤄짐(중복 방지, 재실행으로 중복
   0건 확인).

6. 텍스트 상자 안 문단도 body.iter(w:p)로 찾아지므로 paraId가 정상적으로 붙는다 -
   본문과 똑같이 병합 가능(다만 이번 세션에 직접 실행 검증은 안 했음, 설계 단계).
```

**★ `add_para_ids()`는 멱등적이어야 함(실측 확인, 항등 테스트에서 발견)**: 이미
`*.anchored.docx`가 있으면 그대로 재사용하고, 절대 다시 만들지 않는다 - 매번 새로
만들면 실행할 때마다 다른 무작위 paraId가 부여돼서 캐싱된 elements JSON의 paraId가
최신 anchored 파일과 어긋나는 버그로 이어짐(§0 원칙 4번 "문서 하나당 파싱은 1번만,
JSON 캐싱"과 직결).

### 자동 번호매김 제목 (`auto_num`)

Docling `_add_heading()`이 Word 자동번호매김 스타일 제목에 번호를 합성해서 붙인다
(`"1 미래 전략"` 등, `numbered_headers` 카운터) - 원본 문단의 리터럴 텍스트에는 이
번호가 없다(실측: 198개 중 3개, 전부 이 패턴). 파싱 시점에 Docling 텍스트와 anchored
문서의 원본 문단 텍스트(`Paragraph(xml, docx_obj).text`)를 비교해서, Docling 텍스트가
원본을 접미사로 가지면 그 차이를 `e["auto_num"]`에 저장한다(없으면 `None`). **쓰기
시점**: 번역문 앞에 `auto_num`이 붙어 있으면 떼고 쓴다(원본 문단엔 번호가 없으므로).
**항등 테스트**도 번호를 뗀 텍스트로 비교한다 - writer가 이미 번호를 떼고 쓰므로 별도
비교 로직 없이 자연히 일치함.

### 표 셀의 loc (docx)

**★ 실측 확인(2026-09-26)**: 표 셀 안 문단도 `paragraph_to_items`에 정상적으로 잡힌다 —
커머스 문서로 직접 테스트(셀 문단에 임시 paraId 부여 → `MsWordDocumentBackend.convert()` →
`paragraph_to_items`에서 그 paraId를 찾아 `RefItem.resolve(doc)`으로 실제 아이템을 얻으니
셀 텍스트("비용 절감" 등)가 정확히 나옴). 그래서 **셀도 일반 요소와 같은 원리로 paraId를
우선 쓰되, 셀은 문단이 여러 개일 수 있으므로 `{"loc": {"paraIds": ["...", "..."]}}`(복수형,
셀 안 모든 문단의 paraId 리스트)를 쓴다** — 일반 요소(문단 하나)는 그대로 단수형
`{"paraId": "..."}`. "Docling 표 순서 == python-docx `doc.tables` 순서"라는 가정 자체가
필요 없어짐. **쓰기 규칙**: 번역문은 `paraIds[0]`(첫 문단)에 쓰고, 나머지 문단은 비운다.

**`{"table_index", "row", "col"}`은 paraId를 못 찾은 경우에만 쓰는 fallback**(빈 문단이라
RefItem이 안 생겼거나, 텍스트박스/중첩 표 등 예외 케이스):
```python
{"loc": {"table_index": 0, "row": 0, "col": 1}}
# 쓰기 시점: cell = anchored_doc.tables[0].cell(0, 1); target_para = cell.paragraphs[0]
```
fallback을 쓰기 전에 검증: (a) Docling이 인식한 표 개수와 `len(doc.tables)`가 같은지,
(b) 각 표의 (0,0) 셀 텍스트가 일치하는지. 불일치하면 그 표는 병합에서 skip하고 사유
`"table_mismatch"`로 기록 — 45개 전체에서 몇 건인지 3번 단계(전체 재스캔)에서 보고.

- **병합 셀은 시작 칸 기준으로 1번만** 기록(Docling이 이미 병합 정보를 주므로 동일 원칙).
- **표 안에 또 표가 있는 경우(중첩 표)는 1차 구현 범위 밖** — 발견되면 스킵하고 한계로 기록.

### skip 규칙 (번역·병합만 제외, 청킹·검색·요약엔 포함 — `page_header`/`page_footer`는 furniture라
파싱 단계에서 아예 제외되므로 여기 없음, 2절 참고)

**파싱 시점에 `e["skip"] = 사유문자열`을 기록해둔다**(None이면 번역 대상) — 5절의 매핑
루프가 이 필드만 보고 바로 건너뛰므로, 판정 로직을 매번 다시 계산할 필요가 없고 "skip
사유별 수"(8절 평가지표)도 이 필드를 집계하면 바로 나온다.
- 공통: `"formula"`, `"picture"`
- docx 전용: `"field_code"`(문단에 `w:fldChar`/`w:instrText`), `"footnote_ref"`(`w:footnoteReference`),
  `"tracked_change"`(`w:ins`·`w:del`)

### 헤더 감지
스타일 기반(`Heading N`/`Title`). 45개 중 43개(95.6%)가 실제 스타일 보유(실측). 예외 2개는
알려진 한계.

---

## 4. pdf 파싱 (`pdf_track/parse.py`)

```python
result = _converter.convert(filepath)   # do_ocr=False, TableFormerMode.FAST
doc = result.document
for item, level in doc.iterate_items():   # ★ doc.texts 대신 iterate_items()로 진짜 문서 순서 순회
    if str(item.label) in FURNITURE:      #   (page_header/page_footer는 여기서 바로 제외)
        continue
    provs = []
    for p in item.prov:                      # ★ prov 전체를 저장(문단이 페이지/단을 넘으면 여러 개)
        page_h = doc.pages[p.page_no].size.height
        bbox = p.bbox if p.bbox.coord_origin.name == "TOPLEFT" else p.bbox.to_top_left_origin(page_h)
        provs.append({"page": p.page_no, "bbox": {...}, "charspan": p.charspan})
    elements.append({..., "loc": {"prov": provs}})   # loc이 리스트를 담을 수 있음
```
**이전 버전과 차이 2가지**: (1) `doc.texts`(평면 리스트, 표 위치가 문서 순서와 안 맞을 수 있음)
대신 `doc.iterate_items()`(진짜 문서 순서, `(item, level)` 튜플 반환 — 실측 확인됨)로 순회 —
docx와 같은 이유(표가 본문 흐름과 다른 위치로 밀리는 것 방지). (2) `item.prov[0]`만 쓰던 걸
**`item.prov` 전체**로 바꿈 — 문단이 페이지나 단(2단 논문)을 넘으면 prov가 여러 개 나오기 때문.

### 표: 두 층 구조 (6절과 동일 원칙)
`doc.tables`의 표 하나 = elements의 `label:"table"` 요소 1개. `table.data.table_cells`를
순회해서 `cells` 리스트를 만들고, 표 전체 텍스트는 마크다운으로 직렬화(6절 참고). **셀
loc은 일반 요소와 같은 모양으로 통일**: `{"prov": [{"page":, "bbox": {...}}]}` — bbox는
`table_cells[].bbox`에서(TOPLEFT로 변환), page는 표 자체의 `prov[0].page_no`에서 가져온다.

### 수식 글꼴 skip (pdf 전용) — Docling `formula` 라벨이 1순위, 글꼴 판정은 보조

진짜 수식은 기본적으로 Docling이 이미 분류해주는 `label == "formula"`로 거른다
(`e["skip"] = "formula"`). 글꼴 기반 판정은 formula 라벨이 못 잡는 인라인 수식·기호를
보조로 잡기 위한 것.

```
^(CMMI|CMSY|CMEX|MSAM|MSBM|EUFM|EUSM|EURM|RSFS|STMARY|WASY|TXSY|.*Math|.*Sym)
```
**★ 실측 확인(2026-09-26, `2401.06326v4.pdf`로 검증)**: 원본 PDFMathTranslate(`converter.py`,
확인 완료)는 **글자(span) 단위**로 이 패턴을 적용하지만, 우리는 **요소(문단) 단위**로
적용한다. 처음 원본 패턴(`CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|
wasy|stmary|.*Sym|.*Math`) 그대로 문단 단위에 적용했더니, `CM[^R]`("CM으로 시작하고
세 번째 글자가 R만 아니면 전부 수식")가 `CMBX10`(볼드)·`CMTI8`(이탤릭) 같은 **본문 강조체**
까지 잡아서, 강조 단어 하나만 있어도 문단 전체가 skip 처리되는 문제가 실측됨(196개 요소 중
151개, **77% 오탐**). 그래서 **(1) `CM[^R]` 계열을 빼고 진짜 수식 전용 글꼴 패밀리만
남겼고**(`CMMI`/`CMSY`/`CMEX`/`MSAM`/`MSBM`/`EUFM`/`EUSM`/`EURM`/`RSFS`/`STMARY`/`WASY`/
`TXSY`), **(2) 요소 단위 판정이므로 "글자 수 대비 매치 비율 ≥ 0.5"일 때만 skip**하도록
기준을 추가했다(같은 샘플에서 196개 중 **11개, 5.6%**로 정상화 — `Assumption 2 (Standard
rate of convergence). ‖ ̂ C_YY - C_YY ‖∞, ...`처럼 실제로 수식 표기가 문단 대부분을 차지하는
경우만 남음). 서브셋 접두어(`"ABCDEF+CMR10"`)는 `split("+")[-1]`로 제거하고 `re.match`
(문자열 시작 기준)로 판정 — `re.search`를 쓰면 `"ArialMT"`가 `"MT"`에 우연히 걸리는 등
오탐이 남는다(실측 확인). 매치되면 **파싱 시점에** `e["skip"] = "math_font"`로 기록
(docx의 skip 필드와 같은 방식, 3절 참고).

### 실측 결과

50개 전체 스캔: 실패 0, `section_header` 0개 문서 0, 요소 수 30개 미만 문서 0. **페이지 넘는
표**: `table.prov` 길이 1 초과(Docling 자체 인식) 0건.

**최초에 "인접 페이지 + 열 개수 일치"라는 느슨한 휴리스틱으로 59건의 분할 의심 사례를
찾았고, 그중 `2404.09358v3.pdf`(7열, 11페이지 연속), `2410.22706v2.pdf`(6~9열, 여러 페이지
연속)를 "실제 사례로 확인됨"이라고 판단했었는데 — 이후 더 엄격한 조건으로 재검증한 결과
**이 판단은 틀렸다.** 두 문서를 직접 열어보니:
- `2404.09358v3.pdf` 62·63·64페이지: 각 페이지에 표 1개 + **서로 다른 번호의 캡션**
  (`"Table S2..."`, `"Table S3..."`, `"Table S4..."`)이 붙어있음 — 부록에 비슷한 실험
  조건별로 같은 모양(7열)의 **완전히 별개인 표**가 연달아 나온 것.
- `2410.22706v2.pdf` 14~19페이지도 동일 패턴(`Table 1`~`Table 9`, 각각 자기 캡션 보유).

즉 둘 다 **페이지를 넘는 표가 아니라, 우연히 같은 열 개수를 가진 여러 개의 독립된 표**였다.
"열 개수 일치"만으로는 이 두 경우를 구분 못 했던 것.

**엄격한 조건으로 재검증**(아래 조건 4가지 전부 만족해야 병합 후보): 59건이 나왔던 17개
문서 전체에 대해 다시 검사한 결과 **병합 후보 0건** — 지금까지 확보한 50개 코퍼스 안에는
Docling이 잘못 쪼갠 "진짜" 페이지 넘는 표가 (적어도 이 조건 기준으로는) 없는 것으로 확인됨.

### 페이지 넘는 표 병합 로직 (방어적으로 구현은 해두되, 지금 코퍼스엔 실제 발동 사례 없음)

**조건**(4가지 전부 만족해야 병합):
1. A = p쪽에서 **본문 순서상 마지막 요소**가 표, B = p+1쪽에서 **첫 요소**가 표
   (`doc.iterate_items()`의 진짜 순서 기준 — 앞의 1번 항목 적용)
2. A의 하단 bbox, B의 상단 bbox가 각각 페이지 끝 근처(여백 기준값 이내)
3. 열 개수가 같거나, 셀 왼쪽 x좌표 정렬이 일치
4. **B에 자기 캡션이 없거나, 있어도 "continued"/"계속" 같은 표기** — ★ 이 조건이
   가장 결정적이었음(위 두 오탐 사례를 정확히 걸러냄 — 둘 다 B에 별개의 새 캡션이 있었음)

```python
def merge_split_tables(elements):
    # 위 4개 조건을 전부 만족하는 연속된 "table" 요소만 하나로 합침
    # 합칠 때: cells를 이어붙이되 row 번호를 이전 표 마지막 row+1부터 다시 매김,
    #          헤더 행은 첫 조각 것만 text에 유지 - B의 반복된 헤더는 dup_of로
    #          표시만 해두고(번역은 첫 조각 결과를 복사) 본문 text에서는 제외
    ...
```
**write-back 자체는 이 로직과 무관하게 항상 문제 없음** — 셀 하나하나가 이미 자기만의
정확한 `(page, bbox)`를 갖고 있어서, 병합 여부와 무관하게 번역문은 셀 단위로 정확한 위치에
그대로 쓰임. 이 로직이 있으면 좋은 건 순전히 **청킹·번역 품질**(헤더 맥락 유지) 때문.

**현재 판단**: 지금 50개 코퍼스에는 이 로직이 실제로 발동할 사례가 없는 것으로 확인됐지만,
비용이 크지 않고 다른 논문(향후 코퍼스 확장 시)에는 있을 수 있으니 방어적으로 구현은
해둔다 — 다만 "학부 프로젝트에 흔한 케이스를 위한 과한 엔지니어링"은 아니라는 점을
인지하고, 우선순위는 낮게 둔다.

---

## 5. 청킹 (`smart_chunker.py`)

### semantic은 검색·요약·번역 전부 LangChain `SemanticChunker`로 통일

표준 구현을 쓰는 게 논문 비교에 유리해서 LangChain `SemanticChunker`(langchain_experimental)로
되돌린다. 이 클래스는 `" ".join()`으로 문장을 재조합해서(`\n`이 공백으로 바뀜) `start_index`가
실제 위치와 어긋나는 오프셋 오류가 있다(레포에서 이미 실측 확인된 문제 — start_index 15 vs
실제 16 사례). 그래서 `start_index`는 안 쓰고 아래 `recover_spans()`로 청크 위치를 복원한다:

```python
def recover_spans(flat, chunks):
    """공백 무시하고 청크 텍스트를 flat_text에서 순서대로 찾아 (start, end) 복원"""
    pos = [i for i, ch in enumerate(flat) if not ch.isspace()]
    squeezed = "".join(flat[i] for i in pos)
    spans, cur = [], 0
    for c in chunks:
        key = "".join(c.split())
        k = squeezed.find(key, cur)
        if k < 0 or not key:
            spans.append(None); continue          # 못 찾으면 None → 통계에 기록
        spans.append((pos[k], pos[k + len(key) - 1] + 1))
        cur = k + len(key)
    return spans
```
**fixed도 같은 방식(또는 `start_index`)으로 `(start,end)` 리스트를 만들어서, 이후 요소 매핑
코드는 두 청커가 동일하게 공유한다.**

**영향**: 기존 docx_track 그리드서치 결과는 폐기 — **사유는 "새 파서(paraId 앵커링, 표 2층
구조)로 바뀌어서"**이지 semantic 구현 변경 때문이 아님(semantic은 원래 쓰던 LangChain
`SemanticChunker`로 되돌아왔으므로 그 부분은 바뀌지 않음, 1절 표에 반영됨).

### fixed/semantic — 요소 단위 채점을 위해 오프셋(span)을 추적함

```python
flat_text = "\n".join(e["text"] for e in elements)   # furniture는 애초에 elements에 없음(위 2절)

fixed_docs = CharacterTextSplitter(separator="", chunk_size=CHUNK_SIZE,
                                    chunk_overlap=0, add_start_index=True).create_documents([flat_text])
fixed_spans = [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in fixed_docs]
# fixed는 add_start_index가 정확함(문제는 SemanticChunker만) - 그대로 씀

semantic_chunks = SemanticChunker(embeddings, number_of_chunks=N).split_text(flat_text)
semantic_spans = recover_spans(flat_text, semantic_chunks)   # SemanticChunker는 오프셋 오류 있어 이걸로 복원

def get_target(elements, owner):
    """owner 키(튜플)로 실제 요소 또는 셀 dict를 찾는다."""
    if owner[0] == "el":
        return elements[owner[1]]
    else:  # ("cell", table_id, cell_id)
        table = elements[owner[1]]
        return next(c for c in table["cells"] if c["cell_id"] == owner[2])

fragments_by_owner = defaultdict(list)   # owner 키 -> [(frag_start, frag_text), ...] 문서 전체 누적

for chunk, span in zip(fixed_docs, fixed_spans):   # semantic도 동일한 (chunk_text, span) 쌍으로 처리
    if span is None:            # recover_spans가 못 찾은 청크 - 건너뛰고 통계만 기록
        span_missing += 1
        continue
    start, end = span
    tagged = []                 # ★ 청크마다 새로 초기화, (owner, frag_start, frag_text)
    for e in elements:
        if e.get("skip"):       # ★ 파싱 때 이미 기록된 skip 사유 - 겹침 계산 전에 바로 건너뜀
            continue
        s, t = max(e["span"][0], start), min(e["span"][1], end)   # ★ 겹치는 구간만
        if s >= t:
            continue
        if e["label"] != "table":
            tagged.append((("el", e["id"]), s, flat_text[s:t]))   # 요소 전체가 아니라 겹친 조각만
            continue
        for cell in e["cells"]:       # ★ 표와 겹치면 cells를 span_abs로 다시 비교
            if cell.get("skip"):      # 숫자/기호만 있는 셀 등, 파싱 때 이미 기록됨(6절)
                continue
            cs, ct = max(cell["span_abs"][0], start), min(cell["span_abs"][1], end)
            if cs >= ct:
                continue
            tagged.append((("cell", e["id"], cell["cell_id"]), cs, flat_text[cs:ct]))

    if not tagged:              # ★ 이 청크에 번역할 조각이 하나도 없으면(전부 skip 등) LLM 호출 안 함
        continue
    # ★ 청크 하나 = LLM 호출 1번: "[1] 조각\n[2] 조각\n..." 형태로 한 번에 보내고 [n]으로 응답 파싱
    prompt = "\n".join(f"[{i+1}] {frag_text}" for i, (_, _, frag_text) in enumerate(tagged))
    parsed = parse_numbered(generate(prompt))   # {1: "번역1", 2: "번역2", ...}
    for i, (owner, frag_start, _) in enumerate(tagged):
        fragments_by_owner[owner].append((frag_start, parsed.get(i + 1, "")))

# 문서 전체 처리 끝난 뒤: 같은 요소(또는 셀)의 조각들을 위치 순서대로 이어붙여 채점은 1회만
for owner, frags in fragments_by_owner.items():
    frags.sort(key=lambda f: f[0])
    full_translation = " ".join(text for _, text in frags).strip()   # 조각 사이 공백으로 이어붙임
    target = get_target(elements, owner)
    # full_translation vs target["text"]를 CometKiwi로 1회 채점
```
**owner 키는 튜플로 구분**: 일반 요소는 `("el", element_id)`, 셀은 `("cell", table_element_id,
cell_id)` — `fragments_by_owner`, 번역 결과 저장, writer, 채점 전부 이 키를 그대로 쓰고,
실제 요소/셀 dict가 필요할 때만 `get_target()`으로 찾는다.

**청크 경계가 요소 하나를 반으로 자르는 경우**(fixed/semantic은 요소 경계를 신경 안 쓰므로
흔함): 앞부분은 청크 A에서, 뒷부분은 청크 B에서 각각 따로 번역되고, `fragments_by_owner`에
같은 owner 키로 누적됐다가 **문서 전체 처리가 끝난 뒤 위치 순서대로 합쳐져서** 요소 하나의
완성된 번역문이 되고, 그걸 요소 텍스트 전체와 비교해 **채점은 한 번만** 한다(청크 조각마다
따로 채점하지 않음).

### 세 청커 크기 맞추기: 문서마다 청크 개수를 같게

전체 평균 토큰 수에 이진탐색으로 맞추는 방식 대신, **문서마다 청크 개수(N)를 맞춘다**:
1. smart(`group_elements`, min 100 / max 400 토큰)가 문서를 N개로 나눈다.
2. semantic은 `SemanticChunker(embeddings, number_of_chunks=N)`으로 나눈다 — 경계 기준값이
   N개에 맞게 자동으로 정해짐.
3. fixed는 `CharacterTextSplitter(separator="", chunk_size=ceil(len(flat_text)/N), chunk_overlap=0)`
   으로 나눈다. **`chunk_overlap=0`은 필수** — 기본값 200을 그대로 두면 겹친 부분이 두 번
   번역돼서 요소별 번역문이 중복됨.

결과표에는 청커마다 실제 청크 수, 평균/최소/최대 토큰 수를 같이 기록한다(개수는 근사로만
맞춰짐).

**검색 그리드서치는 이 방식과 별도로 돌린다** — fixed는 chunk_size 3개, semantic은
percentile 3개로. 보고서에는 "청크 개수를 맞춘 본 비교"와 "각 청커 최적값 비교"를 둘 다
싣는다.

### smart — `group_elements()`
```python
def group_elements(elements, count_tokens, min_tokens=100, max_tokens=400, window=2):
    # 구조 우선: section_header 경계에서 새 그룹, 표+캡션(caption_ids로 연결)은 한 그룹
    # 표 하나가 max_tokens를 넘으면: 행 묶음으로 나누고, 나뉜 조각마다 헤더 행을 반복해서 앞에 붙임
    #   ★ 단, 두 번째 조각부터 반복되는 헤더 행은 번역 프롬프트에 "참고 문맥"으로만 넣고
    #     [n] 번호는 안 붙임(이미 첫 조각에서 번역됨 - 중복 번역 방지)
    # 의미 분할: 그 외 그룹이 max_tokens 넘으면 요소 경계(중간 아님)에서 좌우 window 유사도 최저점 분할
    # 크기 조절: min_tokens 미만 그룹은 이웃과 병합
    return [{"element_ids": [...], "text": "..."}]
```

### 세 청커 공정성 (중요)
- 세 청커 다 **정확히 같은 elements 집합, 같은 `flat_text`**에서 출발(furniture는 애초에
  elements에 없음, formula/필드코드 등 skip 대상은 청킹엔 포함).
- 표의 `text`(마크다운 직렬화)는 파싱 시점에 한 번 정해지고, 세 청커 다 같은 표현을 씀 —
  fixed/semantic은 표 중간에서 잘릴 수 있음(이게 비교 포인트, 의도된 것). smart만 표를 안 자름.

### 어블레이션: semantic+size (선택, 우선순위 낮음)

레포의 `srt/splitters.py`에 있는 커스텀 `SemanticTextSplitter`(오프셋 보존판, srt 트랙에서
계속 쓰는 것)에 smart와 같은 `min_chunk_tokens=100`/`max_chunk_tokens=400`을 줘서
"semantic+size"라는 4번째 설정을 추가 비교한다. 비교 순서: fixed → semantic → semantic+size
→ smart. 이렇게 하면 smart가 이기는 게 "크기 제한" 덕인지 "구조 인식(제목·표·캡션)" 덕인지
나눠서 볼 수 있다.

---

## 6. 표: "표 요소 1개 + 셀 목록" 두 층 구조

셀을 낱개 요소로 펼치면 셀 간 관계·헤더·병합·캡션 연결이 사라진다(지난번 발견한 문제).
그래서 표는 항상 이렇게 저장:

```python
{"id":.., "label": "table", "table_id": 0, "caption_ids": [..],
 "n_rows":.., "n_cols":.., "text": "<마크다운 표>",   # 세 청커 공통 직렬화
 "cells": [
    {"cell_id":.., "row":.., "col":.., "row_span":.., "col_span":.., "header": bool,
     "text": "..", "span_in_table": (start,end),  # 표 text 문자열 안에서의 위치
     "span_abs": (start,end),                      # flat_text 전체 기준 절대 위치
     "loc": {...}},   # pdf: {"prov":[{"page":,"bbox":}]} / docx: {"paraIds":[...]} 우선, table_index fallback(3절)
    ...
 ]}
```
병합 셀은 한 번만 기록(Docling `TableCell`의 시작 행·열 + span 사용, docx도 python-docx가
이미 병합을 해석해서 주므로 동일 원칙).

**청킹·검색·요약**: 표 요소의 `text`(마크다운) 사용. **번역·채점·병합만 셀 단위**, 숫자/기호만
있는 셀은 **파싱 시점에** `NUM_ONLY = re.compile(r"[\d\s.,%+\-−×/:()$€₩~]*")`로 `cell["text"]`
전체를 `fullmatch` 판정해서 `cell["skip"] = "num_only"`로 기록해둔다(청킹·검색 때는 그대로
쓰고, 번역 매핑 루프에서 이 필드만 보고 바로 건너뜀 — 5절 참고). **어느 위치 필드를 쓰는지는
청커에 따라 다르다**:
- **smart**: 표 전체를 한 그룹으로 다루므로 그룹 내부에서 셀을 찾을 때 `span_in_table`(표
  자체 텍스트 안에서의 위치)만 있으면 충분 — `flat_text` 절대 위치는 필요 없음.
- **fixed/semantic**: 청크 경계가 표 중간에서 생길 수 있어서, 청크의 `(start,end)`(flat_text
  절대 좌표)와 비교해야 하므로 `span_abs`를 쓴다(5절 코드 참고).

선택 사항으로 표 전체를 "참고용, 번역 금지" 문맥으로 프롬프트 앞에 붙여 셀 번역에 맥락 제공
가능. 병합(smart만)은 각 셀의 `loc`에 씀.

---

## 7. 병합 (`docx_track/writer.py`, `pdf_track/writer.py` — smart 번역만)

### docx

**★ 실측 확인(2026-09-26, 항등 테스트에서 발견) — 애초 설계(run.text 대입 +
iter_inner_content)가 그룹 도형을 파괴하는 버그가 있었음**: 그룹 도형(`wgp`)이 있는
문단(예: 인용구 카드처럼 텍스트박스 여러 개를 그룹으로 묶은 도형을 호스팅하는 "앵커
문단")의 run 중 하나는 텍스트가 아니라 `w:drawing`/`mc:AlternateContent`만 담고
있는데, 그 run에 `.text = "..."`를 대입하면 python-docx가 run 내용 전체를
지우고(clear_content) 새 `w:t`만 넣어서 **도형 자체가 삭제된다** — 통제 실험으로
확인(이 앵커 문단 딱 1개만 되돌려 써도 재현됨: 문단 356개 중 12개, `w:drawing` 1개
소실). 그룹 도형을 여러 개 포함한 문서 전체를 쓰면 도형이 여러 개 사라짐(34개 문단
소실, 실측). **해결**: run 단위가 아니라 **문단에 직접 속한 `w:t`만** 고친다 -
조건: `t.iterancestors(w:p)`의 첫 번째가 이 문단 자신일 때만(중첩된 도형/텍스트박스
안 `w:t`는 그 안쪽 문단이 따로 처리하므로 제외). 이러면 run의 서식(`rPr`)과 그
안의 `w:drawing` 등 다른 자식은 건드리지 않아 안전하고, 부수 효과로 굵게/기울임
run이 그대로 보존된다(8절 "달라진 정도만 기록" 목표를 넘어 완전 보존으로 실측 확인).

```python
doc = DocxDocument(anchored_path)   # ★ 원본이 아니라 *.anchored.docx를 열어야 함(paraId가 거기만 있음)

def direct_text_nodes(para_xml):
    """para_xml에 직접 속한 w:t만 반환 - 조건: t.iterancestors(w:p)의 첫 번째가
    이 문단 자신일 때만(중첩된 도형/텍스트박스 안 w:t는 제외, 위 실측 참고)."""
    return [t for t in para_xml.iter(qn("w:t"))
            if next(t.iterancestors(qn("w:p")), None) is para_xml]

for owner, translated_text in translations.items():   # owner 키는 5절과 동일 - ("el",id) 또는 ("cell",table_id,cell_id)
    target = get_target(elements, owner)
    loc = target["loc"]
    if target.get("auto_num") and translated_text.startswith(target["auto_num"]):
        translated_text = translated_text[len(target["auto_num"]):]   # Word 자동번호는 원본 문단에 없음(3절 auto_num)
    if "paraId" in loc:                       # 일반 요소(문단 하나) - 대부분 이 경로
        target_paras = [find_paragraph_xml_by_paraId(doc, loc["paraId"])]
    elif "paraIds" in loc:                    # 셀(문단 여러 개 가능) - paraIds[0]에 쓰고 나머진 비움
        target_paras = [find_paragraph_xml_by_paraId(doc, pid) for pid in loc["paraIds"]]
    else:                                      # table_index fallback (드문 경우, 3절 참고)
        cell = doc.tables[loc["table_index"]].cell(loc["row"], loc["col"])
        target_paras = [p._p for p in cell.paragraphs]
    for i, para_xml in enumerate(target_paras):
        text_for_this_para = translated_text if i == 0 else ""   # 첫 문단에만 번역문, 나머진 비움
        t_nodes = direct_text_nodes(para_xml)
        if not t_nodes:
            log_no_text_node(owner); continue   # w:t가 아예 없는 문단(빈 문단 등) - 건너뜀
        t_nodes[0].text = text_for_this_para
        t_nodes[0].set(qn("xml:space"), "preserve")   # 앞뒤 공백 보존
        for extra in t_nodes[1:]:
            extra.text = ""
doc.save(f"results/docx/{name}_translated.docx")
```
**알려진 한계**: 문단에 하이퍼링크가 있었는데 그 안의 `w:t`를 비우기만 하면 **빈
하이퍼링크(클릭해도 아무 텍스트 없는 링크)가 남을 수 있음** — 링크 자체를 제거하는
것까지는 1차 구현 범위 밖, 한계로 기록만 함.

### pdf — 페이지마다 정해진 순서로 처리(bbox 겹침 시 방금 쓴 걸 지우는 사고 방지)

```python
CSS = "@font-face {font-family: NotoSansKR; src: url(NotoSansKR-Regular.ttf);} " \
      "body {font-family: NotoSansKR;}"   # 실제 쓰기와 test_fit이 반드시 같은 CSS를 씀

def flatten_targets(elements, translations):
    """표는 셀 단위로 펼쳐서, 일반 요소와 셀을 같은 모양의 (owner, bbox, safe_text) 리스트로 만든다
    - 페이지별로 묶으려면 표 요소 자체가 아니라 그 안의 셀 하나하나가 각자 bbox/page를 갖고
    있어야 하므로."""
    out = []
    for owner, translated_text in translations.items():
        target = get_target(elements, owner)
        bbox = target["loc"]["prov"][0]["bbox"]      # 1차 규칙: 여러 칸 걸치면 첫 칸만(7절 하단 참고)
        page_no = target["loc"]["prov"][0]["page"]
        out.append({"owner": owner, "page_no": page_no, "bbox": bbox,
                    "safe_text": html.escape(translated_text)})   # ★ escape는 여기서 한 번만
    return out

targets = flatten_targets(elements, translations)
for page_no, page_targets in group_by_page(targets):   # page_no 기준 그룹핑(셀도 이미 펼쳐져 있음)
    page = pdf[page_no - 1]   # page_no는 1부터 시작
    scratch_page = ...   # 빈 임시 페이지(항상 재사용)
    # (a) 먼저 넣어보고 들어가는지 확인 - ★ test_fit도 실제 쓰기와 똑같은 CSS·이스케이프된 텍스트 사용
    fits = [t for t in page_targets
            if test_fit(scratch_page, t["bbox"], t["safe_text"], css=CSS)[0] >= 0]
    # (b) 들어가는 것만 전부 redact 표시
    for t in fits:
        b = t["bbox"]
        page.add_redact_annot(fitz.Rect(b["l"] + 1, b["t"] + 1, b["r"] - 1, b["b"] - 1))  # 1pt 안쪽
    # (c) 한 번에 적용
    page.apply_redactions(images=PDF_REDACT_IMAGE_NONE, graphics=PDF_REDACT_LINE_ART_NONE)
    # (d) 그다음에 전부 쓰기
    for t in fits:
        b = t["bbox"]
        rect = fitz.Rect(b["l"] + 1, b["t"] + 1, b["r"] - 1, b["b"] - 1)
        spare_height, scale = page.insert_htmlbox(rect, t["safe_text"],
            css=CSS, archive=pymupdf.Archive("fonts/"), scale_low=0.6)
        # ★ 반환값은 (spare_height, scale) 튜플. spare_height<0이면 아무것도 안 쓰인 것(실패)
        if spare_height < 0:
            log_failed(t["owner"]); continue   # 원문 유지
        log_scale(t["owner"], scale)
pdf.subset_fonts()   # 맨 마지막에 한 번
pdf.save(f"results/pdf/{name}_translated.pdf")
```
**`test_fit`은 실제 `insert_htmlbox` 호출과 완전히 같은 CSS·이스케이프된 텍스트로 시험
삽입한다** — 다르게 하면(예: escape 전 텍스트로 시험) `&`가 `&amp;`로 늘어나는 것 같은
길이 차이 때문에 "들어간다"고 판정했는데 실제 쓰기에선 넘치는 불일치가 생길 수 있음.
**여러 칸(prov 여러 개)에 걸친 문단**: 1차 구현은 **첫 번째 칸(`loc["prov"][0]`)에만 쓰고
나머지 칸은 원문 유지**(charspan 값이 항상 신뢰 가능한지 미확인 — `document.py:4642`에
`charspan=(0,0)` placeholder로 쓰는 코드가 있는 걸 봐서 항상 의미있는 값은 아닐 수 있음).
**여러 칸에 걸친 요소 수는 병합 지표에 별도로 기록**한다. charspan 비율로 쪼개 배치하는 건
스트레치 골로 미룸.

---

## 8. 평가

| 대상 | 방법 |
|---|---|
| 검색 | doc_hit@5/section_hit@5, MRR, F1 |
| 요약 | QA 커버리지(Gemma O/X) |
| 번역 | **CometKiwi, 요소(표는 셀) 단위 — 세 청커 전부**(fixed/semantic 포함, 1절 원칙 2번). `.venv-eval`에서 저장된 (원문,번역) 쌍 읽어서 채점. 실제 작동 확인됨(테스트 점수 0.888). **결과표에 청커별 "채점 요소 수 / 전체 요소 수"를 같이 기록**해서 커버리지를 보임 — `span_missing`(recover_spans가 못 찾은 청크) 등으로 빠진 요소는 채점 실패가 아니라 **커버리지 손실로 별도 보고** |
| 병합 | **항등 테스트가 최우선**(파싱 직후 실행, 통과해야 다음 단계) — 기준은 아래 참고. 그다음 번역 커버리지, 잘린 요소 수, 번호 누락 수, pdf 축소 배율/실패 수, skip 사유별 수 |
| 공통 | 청크 수, 평균 토큰 수, 청킹 시간 |

CometKiwi는 `.venv311`(번역 실행) → 결과 파일 저장 → 프로세스 종료 → `.venv-eval`(채점)
순서로 완전히 분리해서 실행(같은 프로세스에서 절대 같이 못 씀).

### 항등 테스트 기준 (구체화)

- **pdf**: (a) 요소 bbox 안쪽을 `page.get_text(clip=bbox)`로 재추출한 텍스트가 원문과 일치,
  (b) 그림·선(도형) 개수가 redaction 전후로 동일(`apply_redactions`가 불필요한 걸 지우지
  않았는지), (c) 모든 요소 bbox **바깥** 영역은 전후 픽셀이 그대로인지(스크린샷 비교 등)
- **docx**: (a) 텍스트가 원본과 완전히 일치(auto_num 뗀 기준, 위 참고), (b) **저장 전후
  `w:p`/`w:drawing`/`w:pict` 개수가 정확히 같아야 통과**(★ 실측 확인 — 그룹 도형이 통째로
  사라지는 버그를 이 기준으로 잡아냄, w:t 직접 수정 방식으로 해결 후 3개 다 전/후 동일
  확인됨) (c) 굵게/기울임 run 개수도 참고로 기록(w:t 직접 수정 방식에서는 서식이 100%
  보존되므로 실측상 항상 차이 0 — 애초 "몇 개나 달라지는지"를 보려 했던 기준이었으나
  결과적으로 완전 보존됨)

---

## 9. 폴더 구조

```
LLM_chunking/
├─ requirements.txt          # .venv311 전체 pip freeze(docling==2.127.0 포함, 내부 API 의존이라 정확히 고정 필요)
├─ requirements-eval.txt     # .venv-eval 전체 pip freeze(unbabel-comet==2.2.7, transformers==4.57.6)
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
| 4 | `group_elements()`로 smart가 N개 청크 생성 → semantic(`number_of_chunks=N`), fixed(`chunk_size=len(flat_text)/N`)로 개수 맞추기 | 표+캡션 안 잘리는지, 큰 표 헤더 반복되는지 육안 확인, 청크 수/평균 토큰 수 결과표에 기록 |
| 5 | 검색·요약(세 청커, semantic은 LangChain `SemanticChunker`) — **docx 그리드서치 재실행 포함**(별도, chunk_size 3개×percentile 3개) | 기존 지표로 비교, "개수 맞춘 비교"+"각자 최적값 비교" 둘 다 보고 |
| 6 | 번역 `[n]`(세 청커, 요소/셀 단위 채점) | 번호 누락 개수 기록 |
| 7 | smart 병합 + 병합 지표 | 커버리지, 잘린 요소 수, skip 사유별 수, 숫자 보존율 |
| 8 | 스트레치(시간 남으면) | 서식 태그화, 번역 재시도, pdf 여러 칸 문단 나누기, 수식 자리표시자, semantic+size 어블레이션 |

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
| pdf 표가 페이지 경계를 넘음(우려) | Docling 표 구조 모델이 페이지 단위 독립 처리(소스 확인) | 처음엔 2개 문서를 "실제 사례"로 오판했었음(부록의 별개 표 시리즈를 페이지 넘는 표로 착각 — 각 표의 고유 캡션을 못 봐서) — **캡션 유무까지 확인하는 엄격한 조건으로 재검증한 결과 실제 사례 0건**(4절). 방어 로직은 구현해두되 우선순위 낮음 |

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
