# DIP Challenge Lab — 프로젝트 2-1
대용량 비정형 텍스트 스마트 청킹·병합 파이프라인

## 문제 정의
LLM은 한 번에 받는 토큰 수에 한계가 있다. 흩어진 수십 개 .docx나
타임스탬프가 포함된 대용량 자막 파일을 단순 글자 수로 자르면 문맥이 끊긴다.
텍스트를 의미 단위(semantic boundary)로 분할(split)하고, 처리(번역/요약) 후
다시 완벽히 병합(merge)하는 자동화 Python 파이프라인을 만든다.

핵심 질문: **단순 글자수 분할과 스마트 청킹이 최종 결과
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
- 파일 처리: python-docx / PYMuPDF 등

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
- 평가용 경량 LLM: Gemma4-12B-QAT-it (24GB에서 로컬 구동)
- 데이터셋: OpenSubtitles, LongBench/SCROLLS, 자체 .docx 묶음
- GPU: Tier 1 (RTX 3090 / A5000). 이 프로젝트는 GPU보다 CPU/메모리/코드 설계가 핵심
