# v2 구조 정리 (2026-09-29)

목표·병합 안 함 방침은 그대로. 이번 작업은 "트랙별 폴더 + 공통 부품은 상위"
구조 정리 - semantic 방식 변경(아래) 하나만 빼고 로직은 안 바꿨다.

## 새 트리

```
src/
  config.py            경로·모델명 상수
  llm.py               Gemma 로드/배치 생성 + setup_models()/setup_embeddings_and_llm() (common.py 흡수)
  chunkers.py           make_fixed / make_semantic / make_smart / recover_spans (smart_chunker.py 흡수)
  retrieval.py          검색 인덱스 + hit@5/MRR/F1 (indexing.py 흡수, docx·pdf 공용)
  summary.py            top-k 요약 + O/X 판정
  translate.py          [[n]] 블록 번역
  comet_score.py        CometKiwi 채점(.venv-eval 전용)
  run.py                 --dataset -> 트랙 pipeline.run(task, chunker, limit, fake_llm) 호출
  docx_track/{loader.py, pipeline.py}
  pdf_track/{loader.py, pipeline.py}
  srt/{loader.py, mapper.py, timestamp_align.py, subtitle_translate.py, pipeline.py}
  longbench/pipeline.py  자리만(CHUNKERS 골격 + TODO)
docs/
  design.md              설계 문서(신규)
  history/                지난 진행 보고서 전부 이관
```

## 삭제

`indexing.py`, `common.py`, `smart_chunker.py`, `visualize.py`, `srt/splitters.py`,
`srt/subtitle_pipeline.py`(→`srt/pipeline.py`), `src/**/.ipynb_checkpoints/`,
`docs/.ipynb_checkpoints/`.

## semantic 방식 변경 (유일한 로직 변경)

- 전: `SemanticChunker` + `min_chunk_size`를 글자/토큰 비율(2.448)로 환산해서 min
  보정. 청크 문자열을 그대로 반환(간접적으로 `recover_spans`를 거치긴 했지만
  min 보정 자체는 글자 수 기준).
- 후: `SemanticChunker`로 경계만 정하고, `recover_spans`로 원문 (start, end)를
  복원한 뒤 **토큰 수를 직접 세서** min 보정(`min_tokens` 미만 청크는 이웃과 합침,
  합친 크기가 `max_tokens` 이하일 때만 - smart의 `merge_small`과 같은 규칙).
  글자/토큰 비율 환산을 없앤 이유: 영어 pdf에서 그 비율로 환산한 min_chunk_size가
  너무 작게 잡히는 걸 실측으로 확인했기 때문.

## 확인 (실측)

### 1. fixed/smart 회귀 확인 - 리팩터링 전 코드(git 이력)와 후 코드를 같은 문서 4개에
지금 다시 나란히 돌려 비교(단순히 예전에 적어둔 표와 비교하는 것보다 엄격한 방식)

| 문서 | 청커 | 청크 내용 100% 일치(전==후) |
|---|---|---|
| 4342398(docx) | fixed | True (n=26) |
| | smart | True (n=27) |
| KCA_Media(docx) | fixed | True (n=15) |
| | smart | True (n=13) |
| 2401.06326v4(pdf) | fixed | True (n=133) |
| | smart | True (n=83) |
| 2401.06740v2(pdf) | fixed | True (n=101) |
| | smart | True (n=59) |

`SmartChunker`/`SmartTextSplitter` 클래스는 주석 빼고 리팩터링 전후 바이트 단위로
동일함(직접 diff 확인) - 위 결과는 그 확인과 일치한다.

(참고: `docs/history/v2_status_2026-09-28.md`에 적힌 smart 숫자(예: 4342398
n=26)는 이 재실행 결과(n=27)와 다르다 - 원인을 찾아 예전 코드를 지금 그대로
다시 돌려봐도 n=27이 나오는 걸 확인했다(위 표의 "전"도 이번에 새로 실행한
결과). 즉 코드 자체는 안 바뀌었고, 그 문서에 적힌 숫자가 그 날 어떤 이유로
다르게 측정됐던 것으로 보인다 - 리팩터링과 무관.)

### 2. semantic 변경 전후 비교 (같은 4개 문서, percentile=90/min100/max400)

| 문서 | 버전 | n | tok_avg | tok_max | min_tokens 미만 |
|---|---|---:|---:|---:|---:|
| 4342398(docx) | 전 | 26 | 218.3 | 393 | 6/26 |
| | 후 | 25 | 227.0 | 393 | 5/25 |
| KCA_Media(docx) | 전 | 18 | 184.4 | 392 | 5/18 |
| | 후 | 18 | 184.4 | 392 | 5/18 |
| 2401.06326v4(pdf) | 전 | 113 | 177.4 | 400 | 34/113 |
| | 후 | 116 | 172.8 | 400 | 32/116 |
| 2401.06740v2(pdf) | 전 | 75 | 182.5 | 397 | 21/75 |
| | 후 | 72 | 190.1 | 397 | 15/72 |

토큰 기준 min 보정으로 바뀐 뒤 min_tokens 미만 청크 비율이 대체로 줄었다(pdf
2401.06740v2: 21/75→15/72). docx 문서 하나(4342398)는 오히려 청크 수가 1개
줄었는데, 합친 청크가 여전히 max_tokens(400) 안에 들어가서 합쳐진 경우.

### 3. srt(Noah, fake LLM - 원문을 그대로 돌려줌)

fixed/semantic/smart 세 청커 모두:
- `start_index` 없는 청크: 0개
- 청크가 원문의 리터럴 슬라이스가 아닌 경우: 0개 (recover_spans 실패 없음)
- 929개 큐 전부 최소 한 청크에 포함됨(빠진 큐 0개)
- `check_timestamp_integrity` 통과(OK)

원본 큐 텍스트와 왕복 후 텍스트 비교: semantic/smart는 929/929 완전 일치.
**fixed는 29/929가 다름** - 예: `"You know how long..."` → `"You know h ow
long..."`. 원인: fixed는 500자 경계에서 단어 중간(`how`)을 그대로 자르고,
`mapper.merge_to_units`가 조각들을 공백으로 이어붙이면서 단어 중간에 공백이
낀다. 이건 리팩터링 버그가 아니라 fixed 청커 고유의 한계(단어 경계 무시)이고,
비교 기준선으로 의도된 동작이라 숨기지 않고 `docs/design.md`의 "한계"에 기록함.

### 4. `run.py --dataset docx --task translate --limit 1 --fake-llm`

`--chunker smart`, `--chunker all` 둘 다 끝까지 정상 실행:
`{'units_total': 137, 'units_translated': 137, 'missing_numbers': 0, 'truncated': 0}`
(fixed/semantic/smart 세 청커 다 137/137, 0 missing) - 기존 실측(docx1, smart,
137/137, missing 0)과 일치.

## 설계상 판단 (사용자 지시에 명시 안 된 부분, 이번에 정한 것)

- `CHUNKERS`는 각 트랙 `pipeline.py`의 `build_chunkers(embeddings, count_tokens)`
  함수 안에서 만든다(파일 맨 위, import 바로 아래) - 모듈 최상단 변수로 두지 않은
  이유: `HuggingFaceEmbeddings`가 모델 로딩을 하기 때문에 import 시점에 바로
  실행하면 `run.py`를 import만 해도 GPU 모델이 로딩된다. 값 자체(500자, min/max
  토큰)는 파일 맨 위 가까이에서 그대로 보이게 했다.
- `run.py`가 `--chunker all`을 넘기면 각 트랙 `pipeline.run()`이 모델을 한 번만
  불러온 뒤 내부에서 fixed/semantic/smart를 순서대로 도는 방식으로 했다(옛
  `run.py`는 모델을 한 번만 불러왔었는데, `run.py`가 매번 새로 트랙을 호출하는
  구조로 바뀌면서 그 효율을 유지하려면 "all" 처리를 트랙 쪽으로 옮겨야 했음).
- srt는 원래 `--fake-llm`을 지원하지 않았다(항상 실제 Gemma로만 실행). 이번
  검증 항목(4번)에 fake LLM 테스트가 필요해서 `srt/pipeline.py`에 `--fake-llm`
  지원을 새로 추가했다(`_setup()`이 fake_llm이면 Gemma 가중치는 안 불러오고
  토크나이저만 불러옴, `_fake_translate()`가 `mapper.build_prompt`/`parse_marked`로
  원문을 그대로 되돌림). `srt/subtitle_translate.py`(실제 번역 로직)는 손대지
  않았다 - fake 경로는 그걸 거치지 않고 `srt/pipeline.py`에서 직접 처리한다.

## 안 한 것 / 다음에 할 일

- `requirements-eval.txt`가 없어서 이번에 `.venv-eval`에서 `pip freeze`로 새로
  만들었다(README가 참조하는 파일이라 필요했음).
- docx/pdf 전체 코퍼스 재검증은 안 했다(지시대로 "전체 코퍼스는 돌리지 마").
- `docs/design.md`의 "튜닝(20%)" 그리드서치는 아직 안 함.
