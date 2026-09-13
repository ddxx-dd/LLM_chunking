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

## 권장 스택 (RFP 명시, 실제 채택 스택은 아래 "현재 구현 상태" 참고)
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
(단, 청크 "크기"를 재는 단위는 토큰으로 바꿨다 — 아래 "청킹 튜닝" 참고. 이건 자르는
기준인 코사인 유사도 자체를 바꾼 게 아니라 크기 제한의 측정 단위만 바꾼 것이라 규칙 위반이 아니다.)

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

---

# 현재 구현 상태 (2026-09-12 기준)

RFP가 제안한 스택(Qwen3-4B/Gemma-3-4B)에서 출발했지만, 실증 비교 끝에 아래처럼 바뀌었다.
"권장 스택"은 역사적 참고용으로 위에 그대로 남겨두고, 실제로 쓰는 건 이 섹션이 최신이다.

## LLM 선택 (실증 비교 결과)
순서대로 비교: Qwen3-4B(고유명사 오역 편향, 예: "Truman"→"트럼") → Qwen3-8B(동일 편향, 기각)
→ Llama-3.1-8B-Instruct(고유명사 오류 0건, 가장 빠름, 단 라이선스 제약) → Gemma-4-E4B-it(Apache 2.0,
양자화 불필요) → Gemma-4-12B-it 8bit(bitsandbytes 이상치분해로 지나치게 느림, 폐기)/4bit(빠르고
품질 좋음) → Gemma-4-26B-A4B(MoE, "활성 파라미터"는 연산량 지표일 뿐 메모리 지표가 아님 — 24GB에서
4bit로도 실제 OOM 확인, 완전 폐기) → **최종: `google/gemma-4-12B-it-qat-q4_0-unquantized`(QAT, 4bit)**.
QAT(양자화 인식 학습) 체크포인트라 일반 사후 4bit보다 chrF/BLEU/F1이 전반적으로 더 나음이 실측 확인됨.
캐시엔 이 QAT 모델과 `BAAI/bge-m3`만 남아있다(나머지 전부 삭제됨).

## 아키텍처
- `config.py`의 `TOKENIZER`가 전체 파이프라인이 쓰는 모델의 단일 소스.
- `llm/client.py: load_llm()`은 `model_id`에 "gemma-4"가 포함되면
  `Gemma4UnifiedForConditionalGeneration` + 4bit `BitsAndBytesConfig`(NF4)로 로드.
  `device_map={"": 0}` 고정 사용 — `"auto"`는 4bit 모델 일부를 CPU로 오프로드하려다
  bitsandbytes가 거부하는 에러를 냄.
- `generate()`는 단건 생성, `generate_batch()`는 여러 messages를 한 번에 배치 생성
  (왼쪽 패딩 표준 사용). 자막 번역 파이프라인(`eval/subtitle_translate.py`)은
  `translate_chunks_batch()`로 기본 **batch_size=8**을 씀 — batch_size별 속도 실측(16청크
  기준, batch=1 대비 배수): 1(365.6초, 1.00배) → 2(211.6초, 1.73배) → 4(131.2초, 2.79배) →
  **8(84.8초, 4.31배)**. 메모리는 batch=1(10.21GB)→8(12.59GB)로 완만하게만 증가(24GB 중
  절반 수준이라 여유 있음). 배치 처리는 순차 처리와 결과 텍스트가 100% 동일하진 않음(13%
  정도 줄바뀜) — GPU 배치 행렬연산의 부동소수점 비결합성 때문이며, 내용이 깨지는 게 아니라
  자연스러운 패러프레이즈 수준 차이임 (버그 아님, 알려진 현상).
- `eval/glossary.py`, `eval/script_check.py`는 삭제됨 — Gemma 계열은 Qwen3-4B가 필요로 했던
  고유명사 용어집/스크립트 검증 재시도 장치가 필요 없음.

## 평가 지표
- 번역: word-F1 대신 **chrF + BLEU**(둘 다 `sacrebleu`) — word-F1이 한국어에 너무 가혹함.
  `chrF2>60`이 문헌상 "좋은 시스템" 기준인데, 이 프로젝트 점수(en2ko ~14-21, ko2en ~33-38)는
  그보다 낮음 — 파인튜닝 없는 프롬프트 기반 EN-KO 번역의 문헌 패턴과 일치, 이상 아님.
- QA(LongBench): word-level F1(SQuAD 스타일 normalize).
- LLM-as-judge는 도입하지 않기로 결정 — non-determinism과 self-preference bias 문헌
  근거(arXiv:2410.21819, 원문 확인) + fairness 원칙과 충돌 우려 때문에 보류.

## 청킹 튜닝
- `semantic_chunking`은 글자 수 대신 **토큰 수**로 크기를 잰다(임베딩 모델 자체의
  `.tokenizer` 사용) — EN/KO 언어별 글자당 토큰 밀도가 달라서 글자수 기준이 부정확했음.
  `min_chunk_tokens`(너무 작은 파편 청크 방지) / `max_chunk_tokens`(무한정 커지는 것 방지)
  두 파라미터 추가. `fixed_chunking`은 규칙대로 글자수 그대로 유지.
- `min_chunk_tokens` 값 64/128/256/512를 실측 비교(About Time, 15청크 미니테스트) —
  512가 chrF/BLEU 최고였으나 표본이 작아 노이즈 가능성 있음. **현재 기본값: 128**
  (`run_docx_compose.py`/`run_longbench.py`/`run_subtitle_translation.py` 공통).
- threshold(`method`/`amount`)와 `min_chunk_tokens`를 합동으로 튜닝하는 실험은 아직 안 함(보류 중).

## 알려진 이슈 (부분 해결됨)
`eval/timestamp_align.py: align_by_overlap`에 원래 신뢰도 임계값이 없었음 — 정렬이 틀려도
항상 최선의(하지만 틀린) 매치를 반환해서, 오정렬된 큐가 스코어링에서 제외되지 않고 엉뚱한
참조와 비교되어 채점됐음. About Time 25.6%, Interstellar 37.4%가 Jaccard 0.7 미만이었는데,
**직접 원문 검증(Tiedemann 2008 "Synchronizing Translated Movie Subtitles" 논문, pypdf로
원문 직접 확인)한 결과 OPUS 정렬 방식이 "1:0 정렬"(대응 없음)을 명시적으로 허용한다는 걸
확인** — 우리 구현엔 이게 빠져있었음. **수정 완료**: 최선 매치조차 실제 시간 겹침이 0이면
(한쪽 자막에만 그 구간이 존재 - 삭제/누락된 장면) 빈 문자열을 반환하도록 변경, 기존
`compare_per_cue()`의 `if p.strip() and r.strip()` 필터가 자동으로 스코어링에서 제외함(다른
파일 변경 불필요). 실측 배제 개수: About Time 35개, Interstellar 33개, Truman 0개(전체 대비
1.7~3%) — 원문 대조로 확인된 진짜 누락(예: Interstellar "Eject." 구간, 한국어 자막에서
127초간 통째로 빠짐)만 정확히 잡아냄.

**중요 발견(48샘플 수동 대조, About Time en2ko)**: 위 수정 후 남은 낮은 점수(Jaccard<0.7)
큐들의 **95~96%는 실제로 내용이 맞게 정렬돼있음** — 짧은 큐의 타이밍 민감성 + 참조 자막이
직역이 아니라 의역/현지화된 것이라 표면 글자 일치가 낮게 나올 뿐. 즉 **낮은 chrF/BLEU 평균의
근본 원인은 정렬 오류가 아니라 평가 방법론 자체의 구조적 천장**(OPUS식 자막-쌍 참조는 직역이
아님 + 한국어 표현의 다양성 + 모델 자체 번역 품질)이라는 게 확인됨 — RFP의 "한계의 솔직한
서술" 항목에 이 발견을 포함할 것. (아직 안 한 것: 간격이 1초 이상이지만 0은 아닌 애매한
케이스에 대한 처리 — 시도는 해봤지만 지표 개선 효과가 미미해서(추정 +0.3 chrF) 보류함.)

## 시도했지만 효과 없었던 것 (재시도 방지용 기록)
- 동적 few-shot 예시 선택(bge-m3 유사도 top-3) — chrF -1.09, 오히려 악화(타 영화 풀 톤 불일치).
- 영화 메타정보(제목/줄거리) 프롬프트 주입 — +0.18~0.19 chrF, 미미한 효과.
- 장르/카테고리 태그(위키피디아 카테고리 등) 프롬프트 주입 — 직접 시도는 안 했지만, 가장 가까운
  선행연구(EN-일본어 비즈니스 대화 문맥인식 NMT, arXiv:2311.11976, 원문 직접 확인)에서 장르/상황
  태그만 추가했을 때 BLEU +0.15로 우리 위 실험과 같은 수준의 미미한 효과 - 도입 안 함.
- 격식체/존댓말 제어(prompt로 formality 지정) — 우리 세팅(로컬 사전학습 instruct 모델 + 순정
  transformers.generate())에 바로 옮겨올 수 있는 검증된 메커니즘을 문헌에서 못 찾음(찾은 유일한
  formality-control 논문은 힌디어 대상 마스크드 토큰 분류 방식이라 전이 불가). 문헌상 실제 효과가
  있었던 건 정적 태그가 아니라 화자 간 관계/대화 맥락(문서 수준)이었는데 그마저 +0.37 BLEU 정도로
  작고, 구현 난이도(문서 수준 화자-맥락 모델링)가 이득 대비 커서 보류.
- vLLM 마이그레이션 — 아키텍처 지원 자체는 있지만(`Gemma4UnifiedForConditionalGeneration`),
  **이 서버의 NVIDIA 드라이버(510.108.03, CUDA 11.6)가 vLLM nightly 빌드(cu129)보다 너무
  오래돼서 import 단계에서 실패**(`undefined symbol: cuTensorMapEncodeTiled`). 구버전 vLLM
  빌드를 시도하지 않는 한 이 환경에선 막혀있음. 격리 venv에 설치했다가 실패 확인 후 완전
  삭제함(오늘 새로 받은 uv 캐시도 정리, 8/29 생성된 기존 캐시는 보존).

## 환경 주의사항
- **GPU(RTX 3090, 24GB)는 물리적으로 다른 워크로드와 공유됨** — 컨테이너 자체는 전용
  (`PID 1 = jupyter-lab`, 로그인 사용자 root 하나뿐)으로 보이지만, `nvidia-smi`에 우리
  프로세스 목록(`ps`)에 없는 다른 PID가 수시로 나타나 GPU 메모리를 크게 잡았다 놓았다 함
  (실측: 한 번은 이것 때문에 우리 배치 테스트가 실제 OOM 남). 배치 크기나 새 라이브러리
  로드 전엔 `nvidia-smi`로 여유를 확인하는 습관이 필요.
- 배치 크기별 GPU 메모리 실측(batch=1~4, 8청크 기준): 1→9.98GB, 2→10.29GB, 3→10.76GB,
  4→11.01GB peak allocated — 배치를 4배로 늘려도 메모리는 ~10%만 증가(효율적).

## 향후 계획 (미착수)
- threshold + min_chunk_tokens 합동 튜닝.
- `align_by_overlap` 신뢰도 기반 제외 로직 추가 여부 결정.
- DOCX 번들 태스크 코퍼스 확장: `superdoc-dev/docx-corpus`(HF, 73.7만 문서, 확정)에서
  영어 문서 확보 + 한국어(734건, 대부분 "forms"라 얇음)는 최대한 수집 + 부족분은 영어 문서를
  이 프로젝트 자체 번역 파이프라인(자막용과는 별도의 간단한 문단 단위 번역 스크립트, 재사용 안 함)으로
  한국어화. 번역투(translationese)가 semantic_chunking의 문장 유사도 패턴에 영향 줄 수 있음 —
  결과 해석 시 번역 여부를 태그로 남길 것.
- Late chunking(arXiv:2409.04701)은 검토 후 배제 결정 — 일반적인 semantic chunking 관행이
  아니고(임베딩 기법이지 경계선택 기법이 아님), bge-m3 컨텍스트 한도(8192토큰, 직접 확인)가
  자막 전체 문서(About Time 16,072토큰, 직접 측정)엔 부족해서 추가 확장 로직이 필요함.
