# 4단계 - 2. group_elements 병합 규칙 보완 + src/pipeline.py 완성 (2026-09-27)

> 요청하신 두 가지 병합 규칙 보완을 `smart_chunker.py`에 반영하고(실측 재검증
> 완료), 그다음 `src/pipeline.py` 하나로 파싱→청킹→검색/요약/번역→(smart+translate만)
> 병합을 잇는 통합 진입점을 완성했다. 구식 `docx_track/retrieval_benchmark.py`는
> `src/legacy/`로 옮겼다. 전체 dry-run(3단계)·실제 LLM 확인(4단계)은 이번 범위 밖 -
> 여기서는 pipeline.py가 예외 없이 끝까지 도는지만 샘플로 확인했다(실측).

---

## 1. group_elements 병합 규칙 보완 (실측 재검증)

### 수정 1 - 전역 병합 규칙: 합친 크기가 max_tokens 이하일 때만 합침

`_merge_small_groups_global`을 다음 규칙으로 재작성(smart_chunker.py:200-233):
작은 그룹은 **합친 크기가 max_tokens(400) 이하일 때만** 다음 그룹 앞에 합치고,
넘으면 이전 그룹과 합치되(역시 max_tokens 이하일 때만), 그것도 넘으면 작아도
그대로 둔다. 표 그룹은 여전히 병합 대상에서 제외.

### 수정 2 - 작은 표는 쪼개지 않고 통째로 이웃과 병합

`_merge_small_tables`를 신규 추가(smart_chunker.py:236-266): min_tokens 미만인
작은 표는 표 마크다운 블록 전체를 앞 그룹(우선) 또는 뒤 그룹에 그대로 붙인다
(합친 크기가 max_tokens 이하일 때만). 표를 쪼개거나 표 마크다운 안에 다른
텍스트를 섞지 않는다는 원칙은 그대로 유지 - 붙이는 것은 항상 표 블록 "전체".
결과는 `_kind="table"`로 계속 표시해 이후 병합에서 보호.

`group_elements()`는 `_merge_small_tables` → `_merge_small_groups_global` 순서로
호출(스펙에 맞춰 표 먼저 처리).

### 실측 재검증 (같은 샘플 2개, 방금 코드로 직접 재실행)

| | N | min | med | max | min_tokens 미만 | max_tokens 초과(단일요소, 허용) | max_tokens 초과(병합 결과, 있으면 버그) |
|---|---:|---:|---:|---:|---|---|---|
| docx 샘플 | **22** | 101 | 213 | 380 | 0건 | 0건 | **0건** |
| pdf 샘플 | **71** | 15 | 262 | 625 | 1건(15 - 양옆이 표라 병합 불가, 설계상 허용된 예외) | 3건(494/625/456 - 전부 단일 요소, 요소 경계 불가침 원칙에 따른 정상 동작) | **0건** |

요청하신 "병합 후 최대 토큰(단일 요소 초과 제외)"은 docx 380, pdf **392**
(단일요소 3건을 뺀 나머지 중 최대) - 둘 다 max_tokens(400) 이하로, **병합이
원인이 되어 max_tokens를 넘긴 사례는 0건**임을 확인했다.

---

## 2. src/pipeline.py 완성

### 구조

```
python pipeline.py --dataset {docx,pdf,srt} --task {retrieval,summary,translate} \
                    --chunker {fixed,semantic,smart} --limit N [--fake-llm] [--reparse]
```

- **파싱**: `docobj.cached_parse`(캐시 재사용) - docx는 `docx_track.parse.parse_docx`,
  pdf는 `pdf_track.parse.parse_pdf`.
- **청킹**: `smart_chunker.group_elements`가 만든 N개에 세 청커를 전부 맞춤
  (`build_chunks()`) - fixed는 `make_fixed_chunks`, semantic은 `make_semantic_chunks`
  (LangChain `SemanticChunker(number_of_chunks=N)` + `recover_spans`), smart는
  `group_elements`가 준 owners를 그대로 씀. fixed/semantic은 새 `docobj.owners_for_span`
  으로 겹치는 요소/셀을 owner 키로 매핑(표는 셀 단위) - 세 청커 전부 같은 owner
  스키마로 귀결되므로 번역/채점 로직이 청커에 무관하게 하나로 통일됨.
- **번역**(`translate_elements`): 청크 하나당 LLM 호출 1번, `"[1] 조각\n[2] 조각..."`
  형태로 보내고 `[n]`으로 파싱. skip 요소/셀은 제외(`docobj.is_skip_owner`), 조각이
  하나도 없는 청크는 LLM을 호출하지 않는다. 같은 owner에 여러 청크의 조각이
  걸치면 등장 순서대로 `" ".join`.
- **병합**(smart+translate 전용): 기존 검증된 `docx_track/writer.py`(`write_element_text`),
  `pdf_track/writer.py`(`write_translations`)를 그대로 재사용 - `merge_skip` 대상 제외.
- **검색/요약**: 이번 단계에서는 "정상 동작 확인" 수준으로만 구현(원 설계서 6절
  "채점 코드는 연결만, 실행은 3·4번 범위에서만"에 따라 - 실제 Hit@k/MRR·QA
  커버리지 채점과 그리드서치는 5단계 몫). 검색은 `indexing.build_retriever`로
  InMemoryVectorStore 구축 후 스모크 질의 1건, 요약은 최상위 청크 top-k를 이어
  붙여 LLM(or fake) 호출 1번.
- **srt**: 새 로직을 안 만들고 기존 `srt/mapper.py`(`build_prompt`/`parse_marked`
  형식과 동일한 `[n]` 태그 규칙 재사용 - `fake_llm`이 두 트랙에 다 통함)와
  `srt/subtitle_pipeline.py`의 `default_subtitle_chunkers`를 그대로 호출만 한다
  (`process_srt`). en2ko 고정 1방향 - 방향 그리드는 5단계 몫.
- **가짜 LLM**(`--fake-llm`): `[n] 조각` 형태를 감지하면 조각마다 `⟦⟧`로 감싸서
  돌려주고(번역 dry-run용), 아니면 summary용 더미 문자열을 반환. 임베딩 모델은
  fake-llm 여부와 무관하게 항상 로드(청킹엔 항상 필요, Gemma만 생략).

### 샘플 스모크 테스트 (실측, `--fake-llm --limit 1`)

| 실행 | 결과 |
|---|---|
| docx / translate / smart | 22청크, LLM호출 22회, missing_numbers 0, 병합 written 192/192(skipped 0) - 결과 docx를 열어 `⟦...⟧` 마크 160개 문단에서 직접 확인 |
| pdf / translate / fixed | 71청크(smart N과 일치), LLM호출 71회, missing_numbers 0 |
| pdf / translate / smart | 71청크, 병합 written 185/failed 54(fit_by_kind: cell 18/45, el 167/9) - 넣기 실패는 기존에 확인된 pdf writer의 알려진 한계(좁은 셀) 범위, 예외 없이 완주 |
| docx / retrieval / semantic | 22청크(smart N과 일치), InMemoryVectorStore 스모크 질의 성공 |
| docx / summary / fixed | 예외 없이 완주 |
| srt / translate / smart | Noah 22청크, LLM호출 22회, 결과 `.srt` 타임스탬프 보존 확인, 소스 자체에 있던 `?`(원본 자막 파일의 기존 특성 - `srt/loader.py`로 원본을 직접 읽어도 동일하게 나옴을 확인해 파이프라인 문제 아님을 확인) 외 정상 |

전부 예외 0건으로 끝까지 돎. **전체 코퍼스(docx 45·pdf 50·srt 5편) 대상 dry-run은
아직 안 돌렸다** - 그건 3단계 몫이라 이번엔 하지 않음.

---

## 3. 정리 - legacy/ 이동

`src/docx_track/retrieval_benchmark.py`를 `src/legacy/retrieval_benchmark.py`로
이동(git mv, 경로 깊이 같아서 내부 `sys.path` 코드 수정 불필요 - import 확인함).
파일 상단에 대체 이유(★ legacy 표시)를 남겼다. `docx_track/loader.py`(Docling
마크다운 로더)는 `visualize.py`가 아직 참조하고 있어 그대로 둠(요청 범위 밖).
`smart_chunker.py`의 구식 `SmartChunker`/`SmartTextSplitter` 클래스 자체는
**옮기지 않음** - `srt/subtitle_pipeline.py`가 지금도 그대로 쓰고 있어서(자막
트랙은 elements 모델이 아니라 정규식 블록 탐지 방식이 여전히 유효한 설계) 옮기면
자막 트랙이 깨진다. "구식 SmartChunker"라는 표현은 그 클래스를 쓰던 **구식
docx 검색 벤치마크 스크립트**(retrieval_benchmark.py)를 가리키는 것으로 해석함 -
이제 최종 결과는 전부 `pipeline.py` 한 경로로 나온다(docx/pdf 검색·요약·번역·병합
기준).

---

## 다음: 3단계(전체 dry-run) 진행 여부 확인 필요

큰 문제 없이 pipeline.py가 완성됨 - 3단계(docx 45개·pdf 50개·srt 전체 × 세
청커, `--fake-llm`로 전수 dry-run + 체크리스트) 진행 요청 주시면 이어가겠습니다.
