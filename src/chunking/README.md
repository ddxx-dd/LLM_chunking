# chunking: pdf·docx 청커 7종

모든 청커는 같은 모양이다 - **평문의 연속 구간 `[(start, end)]`**를 돌려준다(청크 텍스트 = `text[start:end]`, 겹침 없음).

```python
from chunking import chunk_document, DocContext, Runtime
ctx = DocContext(doc, Runtime.load(), chars_per_token=3.4)   # doc = 전처리된 공통 JSON
spans = chunk_document("B", ctx, p=90)
```

| 청커 | 파일 | 단위 | 크기 조절 값 | 설명 |
|---|---|---|---|---|
| fixed | `textual.py` | 평문 글자 | `T`(목표 토큰) 또는 `size_chars` | LangChain CharacterTextSplitter |
| recursive | `textual.py` | 평문 글자 | `T` 또는 `size_chars`(상한) | LangChain RecursiveCharacterTextSplitter |
| header | `textual.py` | 평문 | (선택) `T` | LangChain MarkdownHeaderTextSplitter(H1~H3), T를 주면 긴 조각만 재분할 |
| semantic | `semantic.py` | 문장 | `p`(백분위) | 이웃 거리가 임계값을 넘는 곳에서 자름 |
| A | `smart_a.py` | 문장 + 구조 규칙 | `p` | semantic + H1 섹션 경계 강제 |
| B | `smart_b.py` | 문장 + 구조 규칙 | `p` (+ beta, decay, w) | 응집도 + 청크당 벌점의 DP 전역 최적, 제목 앞 보너스 |
| E | `smart_e.py` | 문장 + 구조 규칙 | `p` (+ delta_sigma, decay) | 중심 병합, 제목 경계 감점 |

`p`가 클수록, `T`가 클수록 청크가 커진다. min/max 크기 제어는 없다(필요하면 `sizectl/`).

## 파일
| 파일 | 내용 |
|---|---|
| `base.py` | `Chunker` 부모와 등록부(`REGISTRY`, `chunk_document`), `DocContext`(평문·조각·제목 수준·단위·임베딩), 구간 도구 |
| `units.py` | 문장 단위와 구조 규칙 3종(heading_glue, table_atomic, caption_glue) |
| `embed.py` | `Runtime`(encode, count_many, 단위 임베딩 디스크 캐시), bge-m3 fp16 로딩 |
| `io.py` | 청크 파일 입출력(`chunk_dataset`), 태그, 청크 dict, 통계, 내용 보존 검사, 글자/토큰 비율 |
| `sizectl/`, `v1/` | min/max 엔진, 실험 1 옛 파이프라인(README 참고). 새 청커는 import하지 않는다 |

## 청커를 추가하려면
`Chunker`를 상속해 `name`, `knob`, `units_kind`와 `chunk(self, ctx, **params)`를 쓰고 `@register`를 붙인 뒤 `chunking/__init__.py`에서 그 모듈을 import한다.
테스트는 `tests/test_chunking/`에 같은 이름으로 둔다(구간이 연속이고 내용을 빠짐없이 덮는지, 단순 구현과 같은 결과인지).

## 고정값 주의
B의 `beta=1, decay=0.5, w=5`, E의 `delta_sigma=1, decay=0.5`와 `lam`·`theta`를 백분위로 정하는 규칙은 논문 근거가 없는 자체 설계 값이다. 튜닝 전 임시 기본값이다.
