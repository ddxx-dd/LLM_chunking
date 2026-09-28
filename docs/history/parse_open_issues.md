# 1단계(parse) 구현 중 발견한 이슈 2건 — 해결됨

> 실측일: 2026-09-26 · 샘플: `4342398_..._Offer_V1.docx`(commerce), `2401.06326v4.pdf`
> `smart_chunk.md` 최종본 기준 1단계(`docx_track/parse.py`, `pdf_track/parse.py`) 구현·실행 중
> 발견됨. 결정된 해결책을 코드에 반영하고 재실행까지 확인 완료.

---

## 1. docx: `paragraph_to_items`가 list_item·서식혼합 문단을 못 잡음 → 해결

**증상(수정 전)**: 샘플 문서 211개 요소 중 72개(34%)가 `loc: null`.

**원인(소스 확인)**: `MsWordDocumentBackend._handle_text_elements()`의 list_item 분기가
항상 빈 `elem_ref`로 일찍 return해서(2216행) `paragraph_to_items` 등록 자체를 못 하고,
서식이 섞인 문단(볼드+보통 혼합 등)도 마찬가지로 안 잡힌다.

**해결**: `MsWordDocumentBackend`를 상속한 `AnchoredWordBackend`(`docx_track/parse.py`)로
`_handle_text_elements` 자체를 감싸서, 호출 전후 `doc.texts` 길이를 비교해 그 호출에서
새로 생긴 텍스트 객체마다 `obj_to_paraid[id(객체)] = 그 문단의 paraId`를 직접 기록한다 —
list_item이든 서식 혼합이든 내부적으로 해당 메서드 한 번의 호출 안에서 `doc.texts`에
append되므로 100% 잡힌다(`paragraph_to_items`가 뭘 등록하고 안 하는지와 무관). self_ref
대신 객체 id()를 키로 쓴 이유: convert() 뒷단에서 furniture 삭제 등으로 self_ref 문자열이
재계산될 수 있어서, 같은 파이썬 객체(id)를 키로 써야 안전함.

**결과(재실행 확인)**: `loc: null` 72개 → **20개** — 남은 20개는 전부 `picture`(원래도
loc이 없는 게 정상, 파라그래프가 아니므로). **텍스트를 가진 요소 기준으로는 실질적으로
0건** — 추가 텍스트 매칭 보정 로직은 필요 없어짐.

---

## 2. pdf: 수식 글꼴 skip 정규식이 논문 본문 대부분을 삼킴 → 해결

**증상(수정 전)**: 196개 요소 중 151개(77%)가 `math_font`로 오탐.

**원인(실측)**: 원본 PDFMathTranslate(`converter.py`, 확인 완료)는 **글자(span) 단위**로
이 패턴을 적용하는데 우리는 **요소(문단) 단위**로 적용했다. 원본 패턴의 `CM[^R]`
("CM으로 시작하고 세 번째 글자가 R만 아니면 전부 수식")가 `CMBX10`(볼드)·`CMTI8`(이탤릭)
같은 **본문 강조체**까지 잡아서, 문단 안에 강조 단어 하나만 있어도 문단 전체가 skip
처리됨.

**해결**:
1. **패턴 축소** — `CM[^R]` 계열 제거, 진짜 수식 전용 글꼴 패밀리만 남김:
   `^(CMMI|CMSY|CMEX|MSAM|MSBM|EUFM|EUSM|EURM|RSFS|STMARY|WASY|TXSY|.*Math|.*Sym)`
   (서브셋 접두어 제거 `split("+")[-1]`, `re.match` 앵커링은 유지)
2. **비율 기준 추가** — 요소 bbox 안 전체 글자 수 대비 매치 글자 수 비율이 **0.5 이상**일
   때만 `e["skip"] = "math_font"`.
3. Docling `label == "formula"`(수식으로 이미 분류된 것)를 1순위로 먼저 걸러내고, 글꼴
   판정은 그 외 인라인 수식·기호를 잡는 보조 수단으로만 사용.

**결과(재실행 확인)**: `math_font` 151개(77%) → **11개(5.6%)**. 걸린 예시:
`"Assumption 2 (Standard rate of convergence). ‖ ̂C_YY - C_YY ‖∞, ..."`처럼 실제로 수식
표기가 문단 대부분을 차지하는 경우만 남음(단독 `"-"` 2건 포함, 수식 기호만 있는
짧은 조각이라 정상 분류).

---

## 참고
- 반영된 코드: `docx_track/parse.py`(`AnchoredWordBackend`), `pdf_track/parse.py`
  (`MATH_FONT_RE`, `_math_font_ratio`, `MATH_FONT_RATIO_THRESHOLD = 0.5`).
- 설계 문서 갱신: `smart_chunk.md` 3절(list_item/서식혼합 해결 기록), 4절(수식 글꼴 패턴
  축소 + 비율 기준 근거).
