# DIP Challenge Lab — 프로젝트 2-1
대용량 비정형 텍스트 스마트 청킹·병합 파이프라인

## 문제 정의
LLM은 한 번에 받는 토큰 수에 한계가 있다. 흩어진 수십 개 .docx나
타임스탬프가 포함된 대용량 자막 파일을 단순 글자 수로 자르면 문맥이 끊긴다.
텍스트를 의미 단위(semantic boundary)로 분할(split)하고, 처리(번역/요약) 후
다시 완벽히 병합(merge)하는 자동화 Python 파이프라인을 만든다.

핵심 질문: **단순 글자수 분할과 의미 기반 분할이 최종 결과
(요약 품질·번역 자연스러움)에 어떤 영향을 미치는가**

## RFP 핵심 수행 업무 4가지
1. 정규식 + python-docx/os/io 로 파일 I/O 시스템 설계   → 전처리(loader)
2. 임베딩 기반 코사인 유사도로 스마트 청킹 알고리즘 개발  → 의미 분할
3. 토크나이저(BPE 등)로 엄격한 토큰 카운팅·로드 밸런싱   → 제약
4. 두 분할 방식이 최종 결과에 미치는 영향 비교          → 평가

## 권장 데이터 셋
- OpenSubtitles - 타임스탬프 처리/병합 테스트용 다용량 다국어 자막
- Longbench / SCROLLS - 긴 문맥처리 평가 벤치마크
- 자체: 연구실 문서 / 강의자료 .docx묶음으로 실제 병합 시나리오 구성

## 관련 자료 / 사이트
- 개념: Langchain Text Splitters 문서 / Pinecone Chunking 가이드
- 토크나이저: Hugging Face Tokenziers / tiktoken
- 파일 처리: python-docx / PYMuPDF

## RFP 최종 산출물 (9절)
- GitHub 리포: 전처리 코드, README(설치·실행·재현 절차). **재현 가능성 필수**
- Tech Report: 배경·관련연구·제안방법·실험설정·결과(정량/정성)·결론과 한계.
  **OOM 등 트러블슈팅 경험 반드시 포함**
- 성과 공유회: 데모 시연 + 발표

우수 결과물의 조건: ① 정직한 비교(동일 조건) ② 명확한 어블레이션
③ 재현 가능한 코드 ④ 한계의 솔직한 서술

## 권장 스택 (RFP 명시)
- 임베딩: BAAI/bge-m3
- 프레임워크: Sentence-Transformers, LangChain, LlamaIndex
- 평가용 경량 LLM: Qwen3-4B / Gemma-3-4B (24GB에서 로컬 구동)
- 데이터셋: OpenSubtitles, LongBench/SCROLLS, 자체 .docx 묶음
- GPU: Tier 1 (RTX 3090 / A5000). 이 프로젝트는 GPU보다 CPU/메모리/코드 설계가 핵심

---

# 절대 바꾸지 말 것 (알고리즘의 본질)

## fixed_chunking — 단순 글자수 분할
글자 수로만 자른다. 문장·단어·표·자막 큐를 **전혀 고려하지 않는다.**
여기에 문장 경계 존중이나 구조 보호를 덧붙이면 '글자수 분할'이 아니게 된다.

## semantic_chunking — 의미 기반 분할
인접 문장의 코사인 유사도로만 자른다. 구조 정보(타임스탬프·표 행)를 쓰지 않는다.
표를 피해가게 하거나 큐 경계에 스냅시키는 후처리를 붙이지 않는다.

## 구조는 청커의 몫이 아니다

---

# 자료구조 — 청킹이 유닛을 중간에 끊어도 어떻게 복원하는가

## 핵심 모델 (`src/core.py`)

```
Doc(name, text, units, fmt, log)
Unit(start, end, kind, meta)   # start/end는 doc.text 안의 문자 오프셋
Chunk(text, start, end)        # 항상 chunk.text == doc.text[chunk.start:chunk.end]
```

**원칙: 청커는 `doc.text`(순수 문자열)만 본다. 유닛(자막 큐, 표 행)의 경계는 전혀 모른다.**
구조 정보(타임스탬프, 표 행 번호 등)는 전부 `Unit.meta`에 텍스트와 분리되어 있고,
청킹이 끝난 뒤 병합 단계에서만 문자 오프셋 겹침으로 다시 매칭한다.

## SRT(자막) 로드 — `preprocessing/loader.py: load_srt`

자막 큐 하나 = `Unit` 하나. `kind="cue"`, `meta={"index":큐번호, "t_start":초, "t_end":초}`.
`doc.text`는 모든 큐 텍스트를 이어붙인 하나의 문자열(유닛 경계 정보 없음).

## 청킹 → 번역 → 병합 흐름 (`pipeline/mapper.py`)

1. **`fragments(doc, chunk)`** — 청크 하나가 어떤 유닛(들)과 겹치는지 문자 오프셋 교집합으로 찾는다.
   반환: `(유닛인덱스 j, 겹치는 구간 시작 s, 끝 e, 유닛 전체 포함 여부 w)`.
   유닛 하나가 청크 경계에 걸리면 **두 개의 청크에서 각각 다른 fragment로 잡힌다.**
2. **`build_prompt(doc, chunk)`** — 그 청크 안의 fragment들을 `[1] ...`, `[2] ...`처럼
   번호 태그를 붙여 LLM에 보낸다. 유닛이 잘렸다면 그 조각(예: 문장 앞부분만)이 번호 하나를 차지한다.
3. **`parse_marked(llm_output, frs)`** — LLM 응답에서 번호별 번역 결과를 다시 뽑아
   `(j, 번역텍스트)` 목록으로 되돌린다.
4. **`merge_to_units(doc, all_translated_pieces)`** — **여기가 핵심.**
   유닛 인덱스 `j`별로 버킷을 만들고, 모든 청크에서 나온 `(j, 텍스트)` 조각을 해당 버킷에 쌓은 뒤
   순서대로 공백으로 이어붙인다. 유닛이 청크 경계에 안 걸렸으면 버킷에 조각 1개,
   걸렸으면 조각 2개 이상이 들어와 자동으로 이어붙여진다.
5. **`write_srt(doc, unit_texts, out_path)`** — 병합된 텍스트를 **원본 유닛의 타임스탬프**(`u.meta["t_start"/"t_end"]`)
   그대로 써서 SRT를 생성한다. 번역이 아무리 잘려도 타임스탬프는 원본과 100% 동일하게 보존된다
   (`eval/subtitle_translate.py: check_timestamp_integrity`가 이걸 검증).

### 예시 (가상 대사, 실제 대본 아님)

원본 자막 큐 하나: `"안녕, 오늘 좀 피곤해 보이네"` (유닛 #5, 0:10~0:14)

고정 글자수 분할이 이 유닛 중간을 잘라 두 청크에 걸치면:
```
청크 A (앞부분): ... "안녕, 오늘 좀"          -> 태그 [3]
청크 B (뒷부분): "피곤해 보이네" ...          -> 태그 [1]
```
각 청크는 **독립적으로** LLM에 번역 요청되고(청크 B는 청크 A의 문맥을 전혀 모름),
각각 `(5, "Hi, today...")`, `(5, "you look tired")` 같은 결과를 낸다.
`merge_to_units`가 유닛 #5 버킷에 이 둘을 순서대로 이어붙여
`"Hi, today... you look tired"`가 되고, 이 텍스트가 **원본 유닛 #5의 타임스탬프(0:10~0:14)** 그대로
최종 SRT에 기록된다. 문맥 없이 잘려서 번역이 어색해질 수 있는데, 이게 바로 고정 글자수 분할의
한계를 그대로 드러내는 지점이다 — 병합이 실패한 게 아니라 의도대로 정직하게 드러난 것.

## DOCX(표) 로드 — `preprocessing/loader.py: load_docx`

표 행 하나 = `Unit` 하나. `kind="row"`, `meta={"table":표번호, "row":행번호, "header":헤더목록}`,
텍스트는 그 행을 JSON 한 줄로 직렬화한 것. 청킹이 표의 행들을 서로 다른 청크로 쪼개거나
top-k 검색에서 일부 행만 뽑혀도, 그 결손을 보정하지 않고 그대로 드러낸다
(`eval/docx_compose.py: extract_row_dicts`가 검색된 청크에 실제로 들어있는 행만 원본 그대로 표로 렌더링).


