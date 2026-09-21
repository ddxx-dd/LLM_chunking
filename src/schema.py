#[설계 원칙]
#- Doc.text  : 청커(Chunker)가 바라보는 유일한 입력 (순수 텍스트 스트림)
#- Doc.units : 병합기(Merger)와 평가기(Evaluator)가 바라보는 원본 문서의 뼈대(Skeleton).
#- Chunk     : 청커의 출력 (청크 텍스트 및 시작/끝 문자 인덱스 오프셋).

from __future__ import annotations  

from dataclasses import dataclass, field
from typing import Any

# ── 1. Unit : 원본 파일의 최소 조각 단위 ──────────────────────
# start : Doc.text 안에서 이 유닛이 시작하는 글자 오프셋
# end   : Doc.text 안에서 이 유닛이 끝나는 글자 오프셋
# kind  : 유닛 종류 ("cue"=자막 1개 대사, "para"=워드 문단 1개, "row"=워드 표 1행)
# meta  : 원본 복원을 위한 메타데이터 딕셔너리
#         * cue  -> {"index": 7, "t_start": 94.8, "t_end": 96.6}
#         * para -> {"style": "Heading 1", "heading": True}
#         * row  -> {"table": 1, "row": 0, "header": ["연산자", "의미"]}
@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)


# ── 2. Doc : loader.py가 반환하는 표준 문서 컨테이너 ───────────
# name  : 원본 파일명 (예: "부산행.srt", "스택.docx")
# text  : 모든 유닛을 줄바꿈('\n')으로 연결한 단 하나의 통합 평문 문자열
# units : Unit 객체들의 리스트 (원본 타임스탬프 및 서식을 이어주는 뼈대)
# fmt   : 파일 포맷 ("srt", "docx", "text")
# log   : 전처리 과정에서 정제/제거된 태그 및 빈 줄 통계
@dataclass(frozen=True)
class Doc:
    name: str
    text: str
    units: list[Unit]
    fmt: str
    log: dict[str, Any] = field(default_factory=dict)


# ── 3. Chunk : 청커의 출력 단위 ──────────────────────────────
# text  : LLM에 전달될 실제 청크 문자열
# start : Doc.text 안에서의 시작 글자 인덱스
# end   : 끝 글자 인덱스
# 불변식: chunk.text == doc.text[chunk.start:chunk.end]
@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int
