"""청커 공통 틀: 청커 부모 클래스와 등록부, 문서 컨텍스트(DocContext), 구간 도구.

모든 청커는 같은 모양이다 - 평문의 **연속 구간 [(start, end)]**를 돌려준다(청크 텍스트 = text[start:end], 겹침 없음).

    from chunking import chunk_document
    spans = chunk_document("B", ctx, p=90)

청커 하나는 파일 하나(또는 같은 계열 하나)이고 `@register`로 등록부(REGISTRY)에 올라간다.
단위(문장) 만들기와 임베딩은 DocContext.units(kind)가 맡고 청커는 경계만 정한다.
  kind "sent"   평문 전체를 문장 정규식으로 나눈 것(구조 규칙 없음)                 -> semantic
  kind "sentg"  제목 붙임 + 표 한 덩어리 + 캡션 붙임 규칙을 적용한 문장 단위          -> A, B, E
"""
import re
from dataclasses import dataclass

import numpy as np

from common.docjson import render

SENTENCE_SPLIT_REGEX = r"(?<=[.?!。])\s+|\n+"      # LangChain SemanticChunker와 같은 설정(줄바꿈에서도 나눔)
_SPLIT = re.compile(SENTENCE_SPLIT_REGEX)


# ---------------------------------------------------------------------------
# 구간 도구
# ---------------------------------------------------------------------------
def sentence_spans(text, start=0, end=None):
    """[start, end) 구간을 문장 (start, end)로 나눈다. 오프셋은 원문 기준이고 각 구간은 앞뒤 공백을 뺀다."""
    end = len(text) if end is None else end
    cuts, cur = [], start
    for m in _SPLIT.finditer(text, start, end):
        cuts.append((cur, m.start()))
        cur = m.end()
    cuts.append((cur, end))
    out = []
    for s, e in cuts:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


def whole_doc(text):
    """문서 전체를 청크 하나로(앞뒤 공백만 뗌). 비면 []."""
    s, e = 0, len(text)
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return [(s, e)] if s < e else []


def groups_from_cuts(n, cuts):
    """경계 번호 i(단위 i와 i+1 사이)의 집합 -> 그룹 [(a, b)] (단위 a..b-1)."""
    groups, a = [], 0
    for i in sorted(cuts):
        groups.append((a, i + 1))
        a = i + 1
    groups.append((a, n))
    return groups


def spans_from_groups(unit_spans, groups):
    return [(unit_spans[a][0], unit_spans[b - 1][1]) for a, b in groups]


# ---------------------------------------------------------------------------
# 문서 컨텍스트
# ---------------------------------------------------------------------------
@dataclass
class Units:
    spans: list          # [(start, end)] 평문 구간
    emb: np.ndarray      # (n, d) float32, 행마다 길이 1
    ntok: np.ndarray     # (n,) 단위의 토큰 수

    def __len__(self):
        return len(self.spans)


class DocContext:
    """청커가 보는 문서 하나. 평문·조각표·제목 수준은 바로 있고, 단위와 임베딩은 필요할 때 만든다(캐시)."""

    def __init__(self, doc, runtime=None, chars_per_token=None):
        self.doc, self.doc_id, self.dataset = doc, doc["doc_id"], doc.get("dataset")
        self.text, self.pieces = render(doc)
        blocks = {b["id"]: b for b in doc["blocks"]}
        # 제목 조각의 시작 위치 -> 제목 수준(# = 1)
        self.head_level = {p.start: int(blocks[p.block_ids[0]].get("level", 1)) for p in self.pieces
                           if p.kind == "heading" and p.block_ids and p.block_ids[0] in blocks}
        self.runtime = runtime
        self.chars_per_token = chars_per_token
        self._glue, self._units = None, {}

    @property
    def glue(self):
        if self._glue is None:
            from chunking.units import caption_glue

            self._glue = caption_glue(self.doc, self.pieces)
        return self._glue

    def units(self, kind):
        """kind 단위의 구간·임베딩·토큰 수. 임베딩은 runtime.encode로(디스크 캐시가 있으면 그것을 먼저)."""
        if kind not in self._units:
            from chunking.units import make_units

            if self.runtime is None:
                raise RuntimeError("임베딩이 필요한 청커인데 runtime이 없다 (DocContext(doc, runtime=Runtime...))")
            spans = make_units(self, kind)
            hit = self.runtime.cache_get(self.dataset, self.doc_id, kind, spans)
            if hit is None:
                texts = [self.text[s:e] for s, e in spans]
                emb = self.runtime.encode(texts) if texts else np.zeros((0, 1), np.float32)
                ntok = np.asarray(self.runtime.count_many(texts), dtype=np.int32)
                self.runtime.cache_put(self.dataset, self.doc_id, kind, spans, emb, ntok)
            else:
                emb, ntok = hit
            self._units[kind] = Units(spans, np.asarray(emb, dtype=np.float32), ntok)
        return self._units[kind]

    def levels(self, spans):
        """단위마다 '그 단위가 제목으로 시작하면 제목 수준(1~), 아니면 0'."""
        return np.array([self.head_level.get(s, 0) for s, _ in spans], dtype=np.int32)

    def size_chars(self, T=None, size_chars=None):
        """글자수 청커의 크기(글자). size_chars가 있으면 그대로, 없으면 T(목표 토큰) × 이 데이터셋의 글자/토큰 비율."""
        if size_chars is not None:
            return int(size_chars)
        if T is None:
            return None
        if not self.chars_per_token:
            raise ValueError("T(토큰)로 크기를 주려면 DocContext에 chars_per_token이 필요하다")
        return max(50, round(T * self.chars_per_token))


# ---------------------------------------------------------------------------
# 청커 부모 클래스와 등록부
# ---------------------------------------------------------------------------
class Chunker:
    name = ""            # 등록 이름 ("A")
    knob = ""            # 크기를 조절하는 값 이름: "p"(백분위) 또는 "T"(목표 토큰)
    units_kind = None    # 필요한 단위 종류: None | "sent" | "sentg"

    def chunk(self, ctx, **params):
        """DocContext -> [(start, end)] 평문의 연속 구간(겹침 없음)."""
        raise NotImplementedError


REGISTRY = {}


def register(cls):
    inst = cls()
    if not inst.name or inst.name in REGISTRY:
        raise ValueError(f"청커 이름이 비었거나 겹친다: {inst.name!r}")
    REGISTRY[inst.name] = inst
    return cls


def chunk_document(name, ctx, **params):
    if name not in REGISTRY:
        raise KeyError(f"모르는 청커: {name!r} (선택: {sorted(REGISTRY)})")
    return REGISTRY[name].chunk(ctx, **params)
