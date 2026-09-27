# 프로젝트 현재 상황 종합 (2026-09-27 기준)

> DIP Challenge Lab 프로젝트 2-1 - fixed/semantic/smart 세 청커가 검색·요약·번역
> 품질과 원본 구조 병합에 미치는 영향을 비교하는 파이프라인. 설계 전문은
> `smart_chunk.md`, 구현 순서는 그 10절 참고. 이 문서는 지금까지의 작업을 한눈에
> 보기 위한 종합 요약 - 각 항목의 자세한 근거는 `docs/` 아래 개별 문서를 참고.

---

## 진행 단계 (10절 기준)

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | docx·pdf parse | **완료** |
| 2 | 항등 테스트 | **완료** |
| 3 | 45개·50개 전체 재스캔 | **완료** |
| 4 | `group_elements()`로 smart N개 청크 → semantic/fixed 개수 맞추기 | **부분 완료** - `group_elements()` 구현·검증 완료, fixed/semantic 개수 맞추기 + 검색 그리드서치 재실행은 남음 |
| 5 | 검색·요약(세 청커) | 미착수 |
| 6 | 번역 `[n]`(요소/셀 단위 채점) | 미착수 |
| 7 | smart 병합 + 병합 지표 | 미착수(항등 테스트용 writer는 이미 구현·검증됨, 그대로 재사용 가능) |
| 8 | 스트레치 골 | 미착수 |

---

## 1단계·2단계: 파싱 + 항등 테스트 — 완료

- **docx**: `AnchoredWordBackend`(paraId 앵커링) + `iterate_items()` 순서 순회 +
  표 2층 구조(`table_paraids`로 표 셀 직속 문단만 제외) + 보충 요소 안전망
  (`source="supplement"`) + `auto_num`(Word 자동번호 분리) + `merge_skip`
  (table_mismatch/no_loc, skip과 분리).
- **pdf**: `iterate_items()` + `prov` 전체 저장 + 표 2층 구조 + 수식 글꼴 skip
  (패턴 축소 + 글자 비율 0.5 기준) + `item.captions` 기반 캡션 연결(pdf는 실제로
  채워짐, docx는 Docling이 지원 안 해서 인접 검사 폴백).
- **docx writer**: `run.text` 대입 대신 **문단에 직접 속한 w:t만 수정**하는 방식
  (그룹 도형을 통째로 지우던 버그 해결) + `w:t`가 아예 없는 문단(세로 병합 셀의
  vMerge 연속 행)에는 새로 만들어서 씀(`_append_text_run`, 진짜 콘텐츠 손실 버그
  해결) + Fallback 쪽 대응 문단에도 자동으로 같이 씀(`fallback_sibling`).
- **pdf writer**: NanumGothic 폰트로 교체(NotoSansKR 특정 파일의 숫자 깨짐 버그
  회피) + 스크래치 페이지를 대상마다 새로 만듦(공유 시 test_fit 오염 버그 해결) +
  spare(높이)뿐 아니라 추출 텍스트 커버리지 95% 기준 추가 + insert 영역은 원래
  bbox 그대로(redact만 1pt 축소) + CSS font-size를 원문 중앙값으로 + 긴 토큰에
  줄바꿈 지점(U+200B) 삽입 + get_text() 호출을 페이지당 1회로 캐싱(속도).
- **검증 방법**: 표시 테스트(`merge_checks.py`, ⟦원문⟧ 마킹) + 항등 테스트
  (자기 텍스트 되돌려쓰기 후 원본과 비교) + pdf는 단어 중심점 기반 재추출 +
  NFKC/결합문자제거/영숫자만 비교로 조밀한 표의 이웃 셀 섞임 방지.

자세한 버그 목록: `docs/parse_open_issues.md`, `docs/identity_test_findings.md`,
`docs/writer_implementation_review.md`.

---

## 3단계: 전체 재스캔 — 완료, 최종 수치

| 트랙 | 통과 | 핵심 지표 |
|---|---|---|
| **docx (45개)** | **45 / 45 (100%)** | 구조 보존 100%, 미기록 0건, 텍스트 일치율 ≥99.5%(대부분 완전 일치) |
| **pdf (50개)** | 9 / 50(3개 기준 동시 통과) | 그림·도형 보존 **50/50 전부**, 텍스트 일치 ≥99% 통과 34개, 원문 보존 ≥99% 통과 14개, **전체 넣기 성공률 95.8%** |

**docx는 3단계 재검증까지 마치고 완전히 해결됨.** 처음엔 도형 많은 문서에서
수백 건씩 텍스트가 안 써지는 문제가 있었는데(Docling `doc.texts`엔 있지만 우리
`table_paraids` 필터가 표 셀 안 도형까지 걸러냄), 필터를 좁히고 나니 대부분
해결됐고, 남은 잔여 불일치(문서당 수십 자)는 진단 결과 **writer 버그**(세로 병합
셀의 빈 문단 처리 오류)였음이 확인·수정되어 **45개 전부 통과**로 마무리됐다.

**pdf의 낮은 통과 수(9/50)는 실제 병합 품질 문제가 아닌 것으로 확인됨**: (c) 지표
(안 쓴 요소 원문 보존율, 평균 73.5%)를 원본 pdf에서도 같은 방식으로 재검증한 결과
31건 중 30건(96.8%)이 **원본에서도 이미 어긋나는 측정 오류**(Docling bbox 자체의
부정확성) — 실제 손상은 1건뿐(3.2%). 그림·도형 보존 100%, 넣기 성공률 95.8%는
이미 확보됨. 낮은 통과 수는 "비율 기준이 엄격함 + 검증 방법의 한계"가 크다.

자세한 내용: `docs/step3_full_scan_report.md`(1차) → `docs/step3_rescan_final_report.md`
(2차, writer 최적화 포함) → `docs/step3_verification_final.md`(최종 진단·수정, 3차 = 최종).

---

## 4단계: smart 청킹 — 부분 완료

`src/smart_chunker.py`에 `group_elements(elements, embed_model, count_tokens,
min_tokens, max_tokens, window)` 구현 완료:
- 구조 우선(section_header 경계로 섹션 분리, 표는 캡션과 함께 통째로 한 그룹)
- 의미 분할(너무 큰 섹션만 요소 경계에서 임베딩 유사도 최저점 분할 - 기존
  `SmartChunker`와 같은 재귀 이분할 알고리즘 재사용)
- 크기 조절(작은 그룹은 이웃과 병합)
- 큰 표는 행 묶음으로 분할 + 헤더 행 반복(반복된 헤더는 owner 연결 안 함 - 중복
  번역 방지)

**샘플 검증 결과**(실측):
- docx 샘플: 30개 청크, 토큰 3~359
- pdf 샘플: 79개 청크, 토큰 6~625, 표 9개 전부 캡션과 함께 온전히 한 그룹 유지
  (max_tokens 초과 3건은 전부 "쪼갤 수 없는 단일 요소"(밀도 높은 수식 문단) -
  설계상 요소 경계를 안 넘는다는 원칙과 일치하는 정상 동작, 버그 아님)

**아직 안 한 것**:
- smart가 만든 N개를 기준으로 `SemanticChunker(number_of_chunks=N)`/
  `CharacterTextSplitter(chunk_size=len(flat_text)/N, chunk_overlap=0)`로
  fixed/semantic 개수 맞추기(공정 비교의 핵심)
- docx 검색 그리드서치 재실행(새 파서 기준으로, `docx_track/retrieval_benchmark.py`)
- "개수 맞춘 비교" + "각 청커 최적값 비교" 결과표 작성

---

## 5~8단계 — 미착수

검색/요약(5), 번역 `[n]` 태깅+요소 단위 채점(6), smart 병합 실행+병합 지표(7,
writer는 이미 구현·검증 완료라 재사용만 하면 됨), 스트레치 골(8) 전부 아직 시작
안 함.

---

## 지금까지 확정된 주요 설계 결정 (요약)

- 번역 채점은 세 청커 모두 요소(표는 셀) 단위, semantic은 LangChain
  `SemanticChunker` + `recover_spans()`로 위치 복원.
- skip(번역·채점 제외) ≠ merge_skip(쓰기만 제외) - table_mismatch/no_loc은
  merge_skip.
- CometKiwi 채점은 `.venv-eval`에서 별도 프로세스로(실제 작동 확인됨, 0.888).
- 원본 데이터 폴더에는 아무것도 안 씀 - 중간 산출물은 `data/processed/`,
  검증 산출물은 `results/checks/`.
- `requirements.txt`/`requirements-eval.txt`는 각 venv의 전체 `pip freeze`로 고정.

## 알려진 한계 (기록만, 구현 안 함)

- 페이지 넘는 표: 50개 코퍼스에서 실제 사례 0건(방어 로직만 구현)
- 서식 쏠림: 번역문 전체가 첫 `w:t`에 들어가서 문단 안 서식이 섞여 있으면 한쪽으로 쏠림
- 빈 하이퍼링크: 문단의 하이퍼링크 run을 비우면 빈 링크가 남을 수 있음
- pdf 여러 칸(prov)에 걸친 문단: 첫 칸에만 씀
- pdf bbox 겹침으로 인한 검증 오차: Docling이 매긴 두 요소 bbox가 최대 7pt 겹치는
  경우 단어 중심점 방식으로도 완전히는 못 피함(실제 병합엔 영향 미미, 검증 지표에만 반영)
- 헤딩 스타일이 없는 docx 2개(45개 중)

---

## 다음 결정 필요

4단계를 마저 진행할지(fixed/semantic 개수 맞추기 + 그리드서치 재실행), 5단계로
바로 넘어갈지, 아니면 다른 우선순위가 있는지.
