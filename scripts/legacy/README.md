# scripts/legacy: 실험 1(옛 API) 전용 스크립트

옛 청커 API(`chunking/v1`, `chunking/sizectl`)를 쓴다. 새 CLI(`scripts/chunk.py`, `scripts/run.py`)가 새 API로 바뀌어도 실험 1을 재현할 수 있게 동결해 두었다.

| 파일 | 내용 |
|---|---|
| `exp1_main.sh` | 실험 1 실행(세 데이터셋 × fixed·semantic·smart). `chunk_v1.py`, `run_v1.py`를 부른다 |
| `chunk_v1.py` | 옛 `scripts/chunk.py` 복사본(경로만 보정) |
| `run_v1.py` | 옛 `scripts/run.py` 복사본(경로만 보정) |
| `run_combined_v1.py` | 옛 합친 인덱스 평가(실험 1 보조). 새 `scripts/run.py --datasets a,b,c`가 대체 |
| `analyze_exp1.py` | 실험 1 결과 정리·분석(부트스트랩, 구조 지표, 예산 맞춘 고정 분할) |
| `compare_size_control.py` | 그리디 vs DP 크기 제어 비교 |
| `proto_coherence.py` | 응집도 DP 프로토타입(새 B 청커의 전신) |

사용 예: `source env.sh && bash scripts/legacy/exp1_main.sh vectara`
