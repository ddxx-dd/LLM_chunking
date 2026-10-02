# 설계 문서 (v2)

## 목표

LLM에 넣기엔 너무 긴 문서를 어떻게 나눠야 하는가? 단순히 글자 수로 자르는
`fixed`와, 의미 경계를 찾아 자르는 두 가지 방식(`semantic`, `smart`) 중 무엇이
검색·요약·번역 품질에 더 좋은 영향을 주는지 실측으로 비교한다.

병합(번역 결과를 원본 구조 그대로 파일에 되돌려 쓰는 작업)은 이번 범위 밖이다 -
아래 "향후 작업" 참고.

## 데이터셋

| 트랙 | 데이터셋 | 형식 | 문서 수 |
|---|---|---|---|
| docx | Allganize RAG 평가셋 | 한국어 .docx | 45개 문서, QA 211개 |
| pdf | Vectara open_ragbench | 영어 arxiv 논문 .pdf | 100개 문서, QA 559개 |
| srt | 영화 자막 5편(EN/KO 독립 제작) | .srt | 영화 5편 × 양방향 |

docx/pdf는 Docling으로 마크다운(제목 `#`, 표 `|---|`)으로 변환한 뒤 한 문서 = 한
`Document`로 다룬다. srt는 큐(자막 한 줄)를 유닛으로 관리한다(`srt/loader.py`).

## 세 청커

셋 다 `src/chunkers.py`에 구현 하나씩만 있고, LangChain `TextSplitter`를 반환한다.
청크는 항상 원문의 리터럴 부분 문자열(`text[start:end]`)이라서 `add_start_index`가
정확한 위치를 계산해준다 - srt의 타임스탬프 매핑(`srt/mapper.fragments`)과 번역의
블록 매핑(`translate.py`)이 이 전제 위에서 동작한다.

- **fixed** (`make_fixed`): LangChain `CharacterTextSplitter` - 글자 수로만 자른다.
  단어/문장 중간을 그대로 자를 수 있다(의도된 비교 기준선 - 실제로 자막 검증에서
  단어 중간이 잘려 재조합 시 공백이 끼는 사례를 실측으로 확인함, 아래 "한계" 참고).
- **semantic** (`make_semantic`): LangChain `SemanticChunker`로 문장 간 임베딩
  유사도가 급격히 떨어지는 지점(경계 후보)을 찾고, `recover_spans`로 원문 위치를
  복원한 뒤(`SemanticChunker`는 `" ".join()`으로 문장을 재조합해서 원래 위치
  정보를 잃는다) 토큰 수 기준으로 후처리한다: `min_tokens` 미만 청크는 이웃과
  합치고(합친 크기가 `max_tokens` 이하일 때만), `max_tokens`를 넘는 청크는 내부에서
  문장 간 거리가 가장 큰 지점을 골라 반복 분할한다.
- **smart** (`make_smart`): 구조 우선 + 의미 분할 + 크기 제어 3단계.
  ① 마크다운 제목(`#`)마다 섹션, 표는 통째로 한 조각.
  ② 섹션이 `max_tokens`를 넘으면 문단 경계 중 임베딩 유사도가 가장 낮은 지점에서
  분할(`max_tokens`를 넘는 문단만 먼저 문장 단위로 쪼갬).
  ③ `min_tokens` 미만인 작은 청크는 이웃과 합침.
  `mode`에 따라 구조 인식 방식이 다르다(`docx`/`pdf`=마크다운 제목·표,
  `srt`=큐 하나가 한 조각, 큐가 다음 큐로 이어지면 그 사이는 사실상 안 자름).

### 데이터셋별 설정값 (각 트랙 `pipeline.py`의 `build_chunkers`)

| 트랙 | fixed | semantic min/max | smart min/max |
|---|---|---|---|
| docx | 500자 | 100 / 400 토큰 | 100 / 400 토큰 |
| pdf | 500자 | 100 / 400 토큰 | 100 / 400 토큰 |
| srt | 500자(strip_whitespace=False) | 128 / 1024 토큰 | 128 / 512 토큰 |

docx/pdf는 아직 튜닝 전 시작값이다(위 4문서 실측 비교 참고). srt는 이전 실험에서
이미 검증된 값을 그대로 가져왔다.

## 작업과 지표

| 트랙 | 작업 | 지표 |
|---|---|---|
| docx/pdf | 검색(retrieval) | hit@5, MRR, F1(정답 문서 내 청크 vs target_answer 단어 겹침) |
| docx/pdf | 요약(summary) | top-5 검색 결과로 Gemma가 답변 → Gemma가 O/X로 정답 포함 여부 판정(커버리지) |
| docx/pdf | 번역(translate) | `[[n]]` 태그로 블록 단위 번역 → CometKiwi(참조 없는 번역 품질 채점, `.venv-eval` 별도 실행) |
| srt | 번역 | 큐 단위 chrF/BLEU/BERTScore(시간 겹침으로 정렬한 참조 자막과 비교) + 타임스탬프 무결성 |

## 비교 방식

1. **튜닝(20%)**: 문서/QA의 약 20%로 `min_tokens`/`max_tokens`/`percentile` 등을
   그리드서치해서 트랙별 설정값을 고른다(아직 미실시 - 위 표는 시작값).
2. **전체 비교(100%, 시간 되면 80% 재확인)**: 고른 설정값으로 전체 코퍼스를 돌려
   fixed/semantic/smart를 같은 조건에서 비교한다. 표 문서만 따로 집계(`context_type
   =="table"`)해서 구조가 복잡한 문서에서 차이가 더 큰지도 본다.

## 한계

- fixed는 단어/문장 중간을 그대로 자른다 - 이건 결함이 아니라 비교 기준선으로
  의도된 것이다(실측: srt round-trip 검증에서 929개 큐 중 29개가 단어 중간이
  잘려 재조합 시 공백이 끼는 걸 확인함 - fixed 특유의 한계를 감추지 않고 그대로
  기록).
- retrieval의 F1은 페이지 번호가 없어서 쓰는 대체 지표(SQuAD word-F1)라 원래
  낮게 나온다(docx 전체 코퍼스 실측 F1=0.037, hit_rate=0.9는 정상 범위).
- pdf 검색 실측은 아직 5문서 샘플만 돌려봤다(전체 100개는 미실시).
- CometKiwi 채점은 스크립트만 작성됐고 아직 실행해보지 않았다.
- docx/pdf 청커 설정값은 튜닝 전 시작값이다.

## 향후 작업

- 병합: 번역 결과를 원본 문서 구조(표 셀 병합 포함) 그대로 `.docx`/`.pdf`에
  되돌려 쓰는 기능. `main`/`wip/step-c` 브랜치에 이미 구현된 적 있음(다른 문서
  모델 기반이라 그대로 가져올 수는 없음) - v2에서는 범위 밖으로 명시적으로 뺐다.
- docx/pdf 청커 설정값 그리드서치.
