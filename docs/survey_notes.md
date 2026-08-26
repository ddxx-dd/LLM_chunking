# 서베이 노트

## 1. 읽은 자료

### 청킹 전략
- Pinecone Chunking 가이드
- LangChain Text Splitters 문서
  CharacterTextSplitter, RecursiveCharacterTextSplitter, SemanticChunker
- LangChain SemanticChunker 구현
  langchain-ai/langchain-experimental, libs/experimental/.../text_splitter.py
  원출처: Greg Kamradt, "5 Levels of Text Splitting" (LangChain docstring에 명시)
  
### 모델·라이브러리
- Hugging Face NLP Course 챕터 1~2 (from_pretrained, 토크나이저)
- Sentence-Transformers 문서
- python-docx 문서

## 2. 청킹 방식의 발전 단계
| 단계 | 방식 | 기준 | 한계 |
|---|---|---|---|
| 1 | 오프셋 슬라이싱 | 글자 수 | 문장 중간에서 잘림 |
| 2 | CharacterTextSplitter | 구분자 + 크기 | 주제를 고려하지 않음 |
| 3 | RecursiveCharacterTextSplitter | 구분자 우선순위 | 위와 동일 |
| 4 | SemanticChunker | 임베딩 유사도 | 임계값 튜닝 필요 |

본 프로젝트는 1단계와 4단계를 비교한다.

## 3. 임베딩 모델 선정: bge-m3

### Qwen3-Embedding-4B 로드 실패

ValueError: The checkpoint you are trying to load has model type `qwen3`
but Transformers does not recognize this architecture.

원인: 서버 transformers 4.46.3, Qwen3는 4.51+ 필요.
4.51은 Python 3.9+ 요구, 서버는 3.8이라 업그레이드 불가.

venv는 패키지만 격리하고 인터프리터는 시스템 것을 심볼릭 링크로
참조하므로 Python 버전 문제는 해결되지 않는다.

토크나이저는 BPE 규칙 파일(vocab.json, merges.txt)만 읽으면 되어
아키텍처와 무관하므로 Qwen3-4B 토크나이저는 정상 로드된다.
토큰 카운팅에는 이를 사용한다.

### 선정 근거

1. MTEB 종합 순위는 검색(retrieval) 태스크 비중이 크지만,
   본 프로젝트의 임베딩 용도는 인접 문장 간 STS다. 태스크가 다르다.
2. Tier 1(24GB)에서 LLM과 동시 로드 시 여유가 필요하다.
3. 어블레이션을 여러 조합 돌려야 하므로 인코딩 속도가
   실험 횟수를 좌우한다.
4. 문장 단위 임베딩이라 긴 컨텍스트 지원은 불필요하다.

## 4. 구현 중 발견한 것

### 4.1 고정 임계값의 한계

고정 임계값은 문서의 유사도 분포와 무관하게 동작한다.
분포가 다른 문서를 만나면 과소/과대 분할이 발생할 수 있다.

| 데이터 | 평균 유사도 |
|---|---|
| 자막(부산행.srt) | 0.541 |
| 문서(혼합코퍼스_A.docx) | 0.510 |

측정한 두 데이터의 평균 유사도는 0.03 차이로 거의 같았다.
표본이 각 1개뿐이라 "문서마다 크게 다르다"고 단정하기 어렵다.

percentile은 문서 내 상대 순위로 판단하므로 분포 차이가 있어도
자동으로 적응한다. 표본을 늘려 실제로 분포 차이가 나타나는지는
어블레이션에서 확인한다.


### 4.2 percentile의 한계

경계가 실제로 존재하지 않아도 항상 하위 amount%를 자른다.
주제가 하나뿐인 문서에서도 그 비율만큼 잘린다.

자막에서 amount=20으로 실험한 결과 88개 청크가 생성되었고,
그중 2~3토큰짜리 청크가 다수 포함되었다.


### 4.3 문장 버퍼링의 부작용

LangChain SemanticChunker는 buffer_size=1을 기본값으로 각 문장에
앞뒤 문장을 붙여 임베딩한다. 지시 표현("그렇게", "이것")이 있는
문장의 벡터를 안정화하는 효과가 있다.

그러나 인접 버퍼끼리 내용이 겹친다.

buffered[1] = s0 + s1 + s2
buffered[2] =      s1 + s2 + s3
                   ^^^^^^^ 3문장 중 2문장 공유

내용의 2/3가 동일하므로 모든 유사도가 상승하고, 절대 임계값이
의미를 잃는다. LangChain이 buffer_size와 percentile을 함께
기본값으로 둔 이유로 보인다.

베이스라인에서는 buffer_size=0으로 두고 어블레이션 변수로 분리했다.


### 4.4 짧은 문장의 오탐

"어느 누구도 전체를 지휘하지 않는데도 그렇게 된다."

"그렇게"가 앞 내용을 가리키는데 문장 단독으로 임베딩하면
지시 대상을 알 수 없어 벡터가 불안정해진다. 같은 주제인데도
앞 문장과의 유사도가 낮게 나와 경계로 오탐된다.


### 4.5 문체가 유사하면 주제가 달라도 유사도가 높음
"제도는 열려 있었으나 준비할 여력을 가진 층은 한정돼 있었던 셈이다."
"꿀벌 무리는 한 마리씩 떼어 놓고 보면 할 수 있는 일이 많지 않다."

주제는 과거제도와 꿀벌로 완전히 다르지만, 둘 다 제약을 서술하는
설명체 문장이라 유사도가 임계값을 넘었다. 임베딩이 주제뿐 아니라
문장 구조와 어조도 반영하기 때문이다.


## 5. docx 전처리 범위

### 현재 구현
python-docx의 doc.paragraphs로 문단 텍스트만 추출한다.

### 표를 제외한 이유
python-docx 공식 문서(Working with Tables)는 임의의 Word 표를
정확히 표현하려면 복잡한 그래프 자료구조가 필요하며, 문서 색인
용도에서는 "더 단순한 자료구조로 근사하는 것이 일반적"이라고
권장하고 있다.

실제로 헤더가 없는 코드 예제 표에서 헤더/데이터 구조 해석이
실패함을 확인했다. 병합 셀 감지는 Issue #232(2015),
#1312(2023), #1442(2024)에서 반복 제기되었으나 깔끔한 API가 없다.

또한 표의 각 행은 형식이 반복되어 서로 유사도가 높기 때문에
의미 분할의 실험 대상으로 적합하지 않다.

### 텍스트박스를 제외한 이유
Word 내부에서 좌표로 배치되어 문단 순서상의 위치를 복원할 수 없다.
억지로 삽입하면 순서가 뒤섞여 의미 분할이 잘못된 지점에서 자른다.

## 6. 어블레이션 후보

발견한 문제들에서 도출된 변수.

| 변수 | 값 | 근거 |
|---|---|---|
| chunk_size | 300 / 512 / 800 | 기본 |
| 임계값 방식 | fixed / percentile / std | 4.1 |
| amount | percentile: 5/10/20, std: 0.5/1.0/1.5 | 4.2 |
| buffer_size | 0 / 1 / 2 | 4.3, 4.4 |
| min_chunk_size | 없음 / 50 / 100 | 4.2 |
| overlap | 0 / 50 / 100 | 문맥 손실 완화 |
| 데이터 종류 | 자막 / 문서 / LongBench | 4.1 |
| top-k | 1 / 3 / 5 | 검색 개수가 정확도에 미치는 영향 |