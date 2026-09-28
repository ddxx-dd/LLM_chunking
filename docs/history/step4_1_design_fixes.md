# 4단계 - 1. 설계 문제 해결 (샘플 1개로 동작 확인, 2026-09-27)

> `smart_chunker.py`의 `group_elements()`를 수정하고, 세 청커 개수 맞추기 유틸리티
> (`make_fixed_chunks`/`make_semantic_chunks`/`section_for_span`)를 추가했다. docx
> 샘플 1개(`4342398_..._Offer_V1.docx`), pdf 샘플 1개(`2401.06326v4.pdf`)로 동작만
> 확인(실측) - 대규모 재스캔은 3번(dry-run) 단계에서.

---

## group_elements 수정

**변경 전**: 섹션(section_header 경계)별로만 작은 그룹을 병합 - 표나 문서 끝에
막혀서 병합할 데가 없으면 아주 작은 그룹(3토큰짜리 등)이 그대로 남았음.

**변경 후**: 문서 전체 기준 전역 병합으로 교체(`_merge_small_groups_global`):
- min_tokens(100) 미만 그룹은 **다음 그룹 앞에** 합침(순서 유지)
- 문서 마지막 그룹이면 **이전 그룹과** 합침
- 표 그룹은 절대 병합 대상이 아님(캡션 포함 표 마크다운에 남의 텍스트가 섞이면 안 됨)
- 양옆이 다 표거나 유일한 그룹이면 작아도 그대로 둠(드문 예외, 아래 실측 참고)
- 단일 요소가 max_tokens를 넘는 건 그대로 허용(요소 경계는 항상 지킴 - 기존 동작 유지)
- 제목 뒤에서 자르지 않는 원칙은 섹션 구성 단계(헤더가 새 섹션을 열되 뒤따르는 본문과
  같은 그룹)에서 이미 지켜지고 있어서 별도로 손대지 않음

**owners 스키마 추가**: 기존 `element_ids`(최상위 요소 id만) 대신 `owners`(번역
매핑에 바로 쓸 수 있는 owner 키 목록 - `("el", id)` 또는 `("cell", table_id,
cell_id)`)로 바꿈. 표는 이제 셀 단위로 owners가 채워지고(전엔 표 자기 id 하나뿐),
큰 표를 행 묶음으로 나눌 때도 각 조각이 자기 행의 셀 owner만 갖고, 헤더 셀·캡션은
첫 조각에만 owner로 연결됨(6절 원리 - 반복 헤더 중복 번역 방지).

### 실측 결과 (샘플 1개씩)

| | 수정 전 청크 수 | 수정 후 청크 수 | min_tokens 미만 |
|---|---:|---:|---|
| docx 샘플 | 30 | **23** | 1/23 (표에 막혀서 병합 불가, 설계상 허용된 예외) |
| pdf 샘플 | 79 | **71** | 1/71 (표에 막혀서 병합 불가) |

전역 병합으로 청크 수가 줄고(더 균일해짐), 남은 소수의 작은 그룹은 전부 "양옆이
표라 병합 불가"라는 설계상 예외 조건에 정확히 해당함(실측 확인 - 예시 owners
출력으로 직접 위치 확인).

---

## 개수 맞추기 유틸리티 추가

`make_fixed_chunks(flat_text, n)`, `make_semantic_chunks(flat_text, embeddings, n)`,
`recover_spans`, `section_for_span(elements, start)`를 `smart_chunker.py`에 추가(5절
설계를 그대로 코드화).

### 실측 결과

| | smart N | fixed 청크 수 | semantic 청크 수 | recover_spans 실패 |
|---|---:|---:|---:|---:|
| docx 샘플 | 23 | 23 | 23 | 0 |
| pdf 샘플 | 71 | 71 | 71 | 0 |

세 청커 다 정확히 같은 개수로 맞춰짐, semantic 위치 복원 실패 0건.
`section_for_span`도 확인됨(예: fixed 청크 시작 위치 450 -> section_id 16,
문서 맨 앞 청크는 아직 헤더가 안 나와서 `None` - 의도한 동작).

---

## 세 청커 입력 통일 — 방향만 확정, 실제 반영은 2단계(pipeline.py)에서

`fixed`/`semantic`/`smart` 전부 `data/processed/*.json`(elements) → `add_spans`로
만든 같은 `flat_text`를 입력으로 쓰는 구조는 위 유틸리티로 이미 성립함. 기존
`docx_track/retrieval_benchmark.py`(Docling 마크다운 로더 + 구식 `SmartChunker`
사용)는 **별도로 패치하지 않고, 2단계에서 만들 `pipeline.py`가 사실상 대체**하기로
함 — 같은 로직을 두 곳에 유지하는 게 오히려 복잡도를 늘린다고 판단(학부 프로젝트
단순함 원칙). 검색 그리드서치 재실행 자체는 5단계("이후") 몫이라 지금은 코드
경로만 정리.

---

## 다음: 2단계(`src/pipeline.py`)로 진행

큰 문제 없음 - 2단계로 진행 요청 주시면 이어가겠습니다.
