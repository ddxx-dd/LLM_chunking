# 2단계(항등 테스트) 진입 전 점검 3건 + 항등 테스트 중 발견한 문제

> 실측일: 2026-09-26 · 샘플: `4342398_..._Offer_V1.docx`(commerce)
> `smart_chunk.md` 10절 순서 2번(항등 테스트) 진입 조건으로 요청받은 점검 3건과, 실제
> 항등 테스트를 실행하며 새로 발견한 문제를 정리한다.

---

## 점검 1. docx 위치 검증 (paraId → 문단 텍스트 비교)

각 요소의 paraId로 anchored.docx에서 python-docx 문단을 찾아 텍스트(공백 무시)를
요소 text와 비교. 표 셀은 paraIds의 문단 텍스트를 이어붙여 cell text와 비교.

**결과**: 198개 중 3개 불일치, 전부 같은 패턴:
```
('el', 21, '4C68F418', 원본 문단='미래 전략',        요소 text='1 미래 전략')
('el', 67, '62C912FF', 원본 문단='비용 절감',        요소 text='2 비용 절감')
('el', 131,'0B32E701', 원본 문단='매출 증가',        요소 text='3 매출 증가')
```
**원인**: Docling `_add_heading()`이 Word 자동번호매김 스타일 제목에 번호를 합성해서
붙인다(`numbered_headers` 카운터, `msword_backend.py`) — 원본 문단의 리터럴 텍스트에는
이 번호가 없다. **위치(paraId)는 정확하고, 텍스트 내용만 의도된 차이** — 버그 아님.

---

## 점검 2. 같은 paraId를 가진 요소 2개 이상 확인

**1차 결과(최상위 요소만 검사)**: 중복 0건.

**재검사(셀까지 포함)**: **6건 발견** — 실제 버그였다. 표(id=185) 셀 6개의 paraId가
별도의 일반 "text" 요소(id 186~191)와 완전히 겹침 — 같은 텍스트("새로운 e커머스
비즈니스", "미래 전략" 등)가 표 셀로도, 일반 문단으로도 두 번 존재.

**원인**: `doc.iterate_items()`가 "표 셀 안 문단은 표와 별개로 top-level 아이템을 안
만든다"는 가정(`smart_chunk.md` 3절 순서 5번, "조상에 w:tbl이 있는 문단은 별도 text
요소로 만들지 않음")이 실제로는 틀렸다 — 표 셀 문단이 표 요소와 별개로 일반 text
아이템으로도 나온다. 이 가정을 믿고 명시적 필터를 안 넣은 게 원인.

**해결(완료)**: `docx_track/parse.py`에서 anchored 문서의 모든 `<w:p>` 중 조상에
`w:tbl`이 있는 것들의 paraId 집합(`table_paraids`)을 미리 만들고, 일반 텍스트 루프에서
`para_id in table_paraids`면 건너뛰도록 수정. 재실행으로 중복 0건, 요소 수 211→205
(중복 6개 제거) 확인.

---

## 점검 3. 설치된 docling 버전 확인

`importlib.metadata.version("docling")` 및 `docling-2.127.0.dist-info` 폴더명 기준
**실제 설치 버전 = 2.127.0** — `smart_chunk.md`가 가정한 버전과 일치.

**별도로 발견한 문제**: `requirements.txt`에 `docling`/`docling-core`/`docling-ibm-models`/
`docling-parse`/`pymupdf`/`langchain_experimental`/`langchain_text_splitters` 등이
**전혀 적혀있지 않음**(python-docx==1.2.0만 있음). RFP 9절 "재현 가능성 필수" 요구사항과
어긋남 — 별도 처리 필요(아직 미반영, 결정 대기).

사용자가 소스에서 본 줄 번호(2329행)와 이번에 확인한 줄 번호(2216행 근처)가 다른 것은
아마 다른 docling 사본(예: GitHub 최신 버전)을 본 것으로 추정 — list_item이
`paragraph_to_items`에 등록되지 않는다는 결론 자체는 실제 실행으로 검증한 것이라 어느
버전 사본을 봤는지와 무관하게 유효함.

---

## 항등 테스트(docx) 중 발견한 문제 2건

### (A) `add_para_ids()`가 멱등적이지 않음 — 해결됨

호출할 때마다 원본을 다시 읽어 **매번 새 무작위 paraId**를 부여하고 있었다. `parse_docx()`
→ `identity_test.py`처럼 같은 문서에 대해 이 함수가 두 번 이상 호출되는 경로에서, 캐싱된
elements JSON의 paraId가 최신 anchored 파일과 완전히 어긋나는 버그로 이어짐(`smart_chunk.md`
§0 원칙 4번 "문서 하나당 파싱은 1번만, JSON 캐싱"과 정면으로 충돌).

**해결**: anchored 파일이 이미 있으면 그대로 재사용하도록 수정(`docx_track/parse.py`).

### (B) 그룹 도형(wgp) 안 문단이 저장 시 사라짐 — 해결됨

문서 전체(191개 요소)를 자기 자신의 텍스트로 되돌려 쓰고 저장하면, **"그룹 도형"(`wgp`)
안 문단 34개가 통째로 사라졌다**(원본 356개 문단 → 저장 후 322개). 사라진 건 전부 이
문서의 **인용구 카드 3개**(Ed Kennedy/Forrester/Harvard Business Review 인용문 — 텍스트
박스 여러 개를 그룹으로 묶은 도형) 안의 문단 — 최신 표현(Choice)과 구버전 호환
표현(Fallback) 양쪽 다.

**원인(가설 검증 완료)**: 그룹 도형을 호스팅하는 "앵커 문단"(예: "변화하는 시대, 변화하는
구매자")의 run 중 하나는 텍스트가 아니라 `w:drawing`/`mc:AlternateContent`만 담고
있는데, python-docx의 `run.text = "..."` 대입은 run 내용 전체를 지우고(clear_content)
새 `w:t`만 넣는다 — 이 run에 텍스트를 쓰면 **도형 자체가 삭제된다**. 검증: 이 앵커
문단(paraId `5CA63396`) **딱 1개만** 되돌려 써도 재현됨(356→344, `w:drawing` 24→23,
해당 그룹의 문단 12개 소실) — 그룹 안쪽 문단만 건드린 최초 실험이 문제없었던 건 단순히
앵커를 안 건드렸기 때문.

**해결**: writer를 run 단위(`run.text`/`iter_inner_content`)에서 **문단에 직접 속한
`w:t`만 고치는 방식**으로 교체 — 조건: `t.iterancestors(w:p)`의 첫 번째가 이 문단
자신일 때만(중첩된 도형/텍스트박스 안 `w:t`는 그 안쪽 문단이 따로 처리하므로 제외).
첫 `w:t`에 텍스트를 쓰고 `xml:space="preserve"`를 설정, 나머지 `w:t`는 비움. `w:t`가
없으면 `"no_text_node"`로 기록하고 건너뜀.

**재실행 결과**: 356 → **356 유지**, `w:drawing` 24→24, `w:pict` 19→19, 텍스트 완전
일치, **부수 효과로 bold/italic run 개수도 완전 보존**(402→402, 17→17 — run의 서식은
안 건드리므로). 항등 테스트에 "저장 전후 `w:p`/`w:drawing`/`w:pict` 개수 동일" 기준을
추가해서 이 버그를 다시 놓치지 않도록 회귀 감지를 걸어둠.

---

## 참고
- 반영된 코드: `docx_track/parse.py`(`table_paraids` 필터, `add_para_ids` 멱등화, `auto_num`
  감지), `docx_track/identity_test.py`(`w:t` 직접 수정 방식, 구조 보존 체크 추가).
- `requirements.txt`/`requirements-eval.txt`를 각각 `.venv311`/`.venv-eval` 기준
  전체 `pip freeze`로 재생성 완료(docling/docling-core/pymupdf/langchain-experimental
  등 실제 import하는 패키지 전부 포함).
- 최종 결과: docx 항등 테스트 **통과**(텍스트 완전 일치, w:p/w:drawing/w:pict 개수
  저장 전후 동일, bold/italic run 개수 완전 보존).
