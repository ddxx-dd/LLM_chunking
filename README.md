# LLM Chunking Pipeline

fixed(글자수 분할) vs semantic(의미 분할) vs smart(구조+의미+크기 분할) 청킹이
검색·요약·번역 품질에 미치는 영향을 비교하는 실험 파이프라인. 설계 배경과 한계는
[`docs/design.md`](docs/design.md) 참고.

## 설치

두 개의 가상환경이 필요하다.

```bash
# 1) 메인 파이프라인(청킹/검색/요약/번역) - Python 3.11
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# 2) CometKiwi 채점 전용(모델 의존성 충돌 방지 위해 분리) - comet_score.py만 여기서 실행
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

## 실행

```bash
source .venv311/bin/activate
cd src
python run.py --dataset {docx,pdf,srt} --task {retrieval,summary,translate} \
              --chunker {fixed,semantic,smart,all} [--limit N] [--fake-llm]
```

- `--chunker all`: fixed/semantic/smart를 모델을 한 번만 불러온 뒤 순서대로 전부 돈다.
- `--limit N`: 문서/영화 수를 N개로 제한(빠른 확인용).
- `--fake-llm`: Gemma 대신 원문을 그대로 돌려주는 가짜 번역 함수를 써서 청킹·매핑
  로직만 빠르게 확인한다(실제 번역 품질은 확인 못 함).
- srt 트랙은 검색/요약 태스크가 없다 - `--task translate`만 지원.
- 결과는 `results/`에 저장된다(문서/청커 조합별로 이미 있으면 건너뜀 - 이어하기).

CometKiwi 채점:

```bash
source .venv-eval/bin/activate
python src/comet_score.py <pairs.json>
```

## 폴더 구조

```
src/
  config.py, llm.py, chunkers.py     경로/모델 상수, Gemma 로드+배치 생성, 공통 청커 3종
  retrieval.py, summary.py, translate.py, comet_score.py   공통 태스크(검색/요약/번역/채점)
  run.py                              CLI 진입점 - 트랙별 pipeline.run() 호출
  docx_track/, pdf_track/             loader.py(Docling→마크다운) + pipeline.py(청커 설정+실행)
  srt/                                자막 로더/청킹/번역/타임스탬프 정렬(완성본)
docs/
  design.md                           설계 문서(이 파일이 가리키는 것)
  history/                            지난 세션의 진행 보고서(참고용, 지금 코드와 안 맞을 수 있음)
data/, results/                       입력 데이터(git 추적 안 함), 실행 결과(git 추적 안 함)
```
