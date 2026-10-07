# v1: 실험 1(옛 pdf·docx 파이프라인) 보존

실험 1(fixed / semantic / smart, min 128·max 512·percentile 85, 500자)을 만든 코드 그대로다. **동작을 바꾸지 말 것**(결과 재현용).
srt 청킹도 아직 이 경로를 지난다(`srt/README.md` 참고).

| 파일 | 내용 |
|---|---|
| `units.py` | 구조 제약 규칙 6개(`Rules`)와 단위 만들기(`build_units`), `caption_glue` |
| `sections.py` | 최상위 섹션 분할(`section_level`, `split_sections`)과 짧은 섹션 병합 |
| `splitters.py` | 청커 디스패처(`chunk`, `chunk_spans`): fixed / semantic / smart |
| `runner.py` | 데이터셋 → 청크 파일, 태그 이름 규칙 |
| `stats.py`, `meta.py`, `ratio.py` | 청크 통계, 청크 메타데이터, 글자/토큰 비율 측정 |

실행하는 스크립트는 `scripts/legacy/`(`chunk_v1.py`, `run_v1.py`, `exp1_main.sh`, `analyze_exp1.py`)에 있고, 테스트는 `tests/test_v1/`, `tests/test_sizectl/`에 있다.
엔진은 `chunking/sizectl/`을 쓴다.
