# v2 폴더 구조 및 각 파일이 하는 일 (2026-09-28 기준)

> 진행 상황·실측 결과는 `docs/v2_status_2026-09-28.md` 참고. 이 문서는 "지금 뭐가
> 어디 있고 뭘 하는지"만 정리한 구조도.

## 전체 트리

```
LLM_chunking/
├── requirements.txt        .venv311 전체 pip freeze(docling/langchain 포함)
├── smart_chunk.md          v2 이전(050a7ef) 시점의 원래 설계서 - Docling 마크다운
│                            로더 기반 구상. 지금 코드와 100% 일치하진 않음(참고용).
├── CLAUDE.md, README.md    프로젝트 개요·실행 방법(구버전 - 갱신 필요)
├── docs/
│   ├── v2_status_2026-09-28.md   ← 진행 상황·실측 결과·미완료 목록
│   ├── v2_architecture.md        ← 이 문서
│   ├── removal_plan.md           (main 브랜치 정리 때 썼던 계획 문서, 참고용)
│   └── history/                  main 브랜치의 병합 단계 보고서 9개 + 구 설계서
│                                  1개를 트러블슈팅 기록용으로 복사해둔 것(v2와 무관)
├── data/
│   ├── allganize/           docx 코퍼스(45개) + QA(rag_evaluation_result.csv)
│   ├── vectara_ragbench/    pdf 코퍼스(50개) + QA(queries/answers/qrels.json)
│   ├── srt_eng/, srt_kor/   자막 원본(영화 5편, 영/한 각각)
├── results/                 실행 결과(청커 그리드서치 산출물 등) - git 추적 안 함
└── src/
    ├── config.py            경로·모델명 상수(ALLGANIZE_DIR, VECTARA_DIR, TOKENIZER 등)
    ├── llm.py                Gemma 로드 + 생성 함수
    ├── common.py             setup_models() - srt 트랙이 아직 씀(임베딩+LLM 한 번에 로드)
    ├── indexing.py           InMemoryVectorStore 래퍼(build_retriever) - retrieval.py가 씀
    ├── smart_chunker.py      세 청커 중 semantic·smart 구현 + srt용 구식 SmartChunker
    ├── retrieval.py          검색 태스크(코퍼스 인덱싱 + hit@5/MRR/F1)
    ├── summary.py            요약 태스크(top-k 검색 → Gemma 답변 → O/X 판정)
    ├── translate.py          번역 태스크([[n]] 태그 매핑 + CometKiwi용 쌍 생성)
    ├── comet_score.py        .venv-eval 전용 - CometKiwi 채점(별도 프로세스)
    ├── run.py                CLI 진입점 - 위 태스크들을 --dataset/--task/--chunker로 실행
    ├── visualize.py          (구버전 시각화 도구 - 지금 파이프라인과 연결 안 돼 있음)
    ├── docx_track/
    │   ├── loader.py         Docling으로 docx → 마크다운 1개 Document로 변환
    │   └── __init__.py
    ├── pdf_track/
    │   ├── loader.py         Docling으로 pdf → 마크다운 1개 Document로 변환(cuDNN 비활성화 포함)
    │   └── __init__.py
    ├── srt/                  자막 트랙(안 건드림) - 로더/청커/번역/타임스탬프 정렬까지 완성본
    │   ├── loader.py, mapper.py, splitters.py, subtitle_pipeline.py,
    │   │   subtitle_translate.py, timestamp_align.py
    └── longbench/            비어있음(미구현, 범위 밖)
```

## 데이터 흐름 (한 문서 기준)

```
docx_track/loader.py 또는 pdf_track/loader.py
        │  (Docling → 마크다운 문자열 1개, langchain Document)
        ▼
smart_chunker.py의 세 청커 중 하나
  - fixed    : langchain CharacterTextSplitter
  - semantic : SemanticMaxSplitter(SemanticChunker + 문단/문장 경계 보정 + max_tokens 후처리)
  - smart    : SmartTextSplitter(SmartChunker) - 마크다운 제목(#)/표 구조를 이용해 분할
        │  (청크 = 원문의 리터럴 부분 문자열, start_index 포함)
        ▼
   ┌────┴─────┬──────────────┐
   ▼          ▼              ▼
retrieval.py  summary.py   translate.py
(인덱싱+검색) (top-k+요약)  ([[n]] 태그로 LLM 번역, comet_score.py가 나중에 채점)
```

`run.py`가 이 전체를 `--dataset {docx,pdf,srt} --task {retrieval,summary,translate}
--chunker {fixed,semantic,smart,all}`로 묶어서 실행한다. srt는 이 파이프라인을 안
타고 기존 `srt/subtitle_pipeline.py`를 그대로 호출한다(이미 완성된 별도 트랙).

## 참고

- `docx_track/`, `pdf_track/`는 이제 `loader.py`(입력 변환)만 남아있다 - 예전에
  있던 `retrieval_benchmark.py`(docx 전용 검색 벤치마크 초안)는 `src/retrieval.py`
  로 흡수·일반화되면서 삭제됨.
- `src/.ipynb_checkpoints/`, `docs/.ipynb_checkpoints/`는 Jupyter가 자동으로 만든
  체크포인트 잔여물 - 코드가 아니라 정리 대상(필요하면 요청 시 삭제).
- `main`/`wip/step-c` 브랜치는 병합(문서 구조 그대로 번역 결과 써넣기)까지 하는
  완전히 다른 버전 - v2는 병합을 안 하고 청킹 전략 비교만 한다.
