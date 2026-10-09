# srt (자막) 패키지

영화 자막(srt) 쌍(영어·한국어)을 다루는 코드를 한곳에 모았다. 번역 실험용 데이터셋이라 pdf·docx 청킹 실험과는 따로 관리한다.

| 파일 | 하는 일 |
|---|---|
| `prep.py` | srt 원문 → 공통 JSON(cue 블록). **Docling이 아니라 정규식 파서**(`parser: regex-srt`: 인코딩 순차 시도, 큐 정규식, 태그 제거, NFC·공백 정리). 영화 쌍 목록(`SRT_MOVIES`), 매니페스트(`manifests/srt.json`), 전처리 대상 목록(`srt_entries`) |
| `validate.py` | 전처리 결과 점검(큐 개수, 시각). 공통 점검은 `preprocess/validate.py`가 먼저 한다 |
| `render.py` | cue 블록 → 평문 + 조각표. `common.docjson.render`가 fmt가 srt이면 여기로 위임한다. `block_spans`, `to_lc_document` 포함 |
| `smart.py` | srt용 smart 청커(옛 `legacy_smart.py`를 이름만 바꿔 이동. 내용 그대로) |
| `config.py` | srt 데이터셋 설정(Gemma 토크나이저, 청커별 파라미터). `common/config.py`가 등록만 한다 |

## srt 청킹이 지나가는 길
srt의 fixed·semantic·smart는 아직 pdf·docx와 같은 옛 실행 경로를 쓴다.
`chunking/v1/runner.py`(실행) → `chunking/v1/splitters.py`(디스패처) → `chunking/sizectl/`(min/max 의미 분할 엔진), `srt/smart.py`.
새 pdf·docx 청커(`chunking/` 바로 아래)는 srt를 다루지 않는다. srt 전략은 나중에 정리한다.

## 이 패키지 밖에 남아 있는 srt 관련 코드
- `common/docjson.py`: `fmt == "srt"`이면 `srt.render`로 넘기는 한 분기
- `common/models.py`: Gemma 토크나이저(`get_token_counter("gemma")`)
- `chunking/v1/meta.py`, `chunking/v1/stats.py`: 청크의 시각(`t_start/t_end`)과 큐 분할 비율
- `tests/test_srt/`, `tests/test_docjson.py`, `tests/test_preprocess.py`의 srt 사례
