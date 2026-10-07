"""청킹 패키지 (pdf·docx용 청커 7종).

    from chunking import chunk_document, DocContext, Runtime
    ctx = DocContext(doc, Runtime.load(), chars_per_token=3.4)
    spans = chunk_document("B", ctx, p=90)          # [(start, end)] 평문의 연속 구간

청커: fixed, recursive, header (textual.py) / semantic (semantic.py) / A (smart_a.py) / B (smart_b.py) / E (smart_e.py).
하위 패키지: sizectl/ (min/max 크기 제어 엔진), v1/ (실험 1 옛 파이프라인 보존). 자막(srt)은 최상위 srt 패키지.
"""
from chunking import semantic, smart_a, smart_b, smart_e, textual  # noqa: F401  (import하면 등록부에 올라간다)
from chunking.base import REGISTRY, Chunker, DocContext, Units, chunk_document
from chunking.embed import Runtime

CHUNKERS = tuple(REGISTRY)

__all__ = ["CHUNKERS", "REGISTRY", "Chunker", "DocContext", "Runtime", "Units", "chunk_document"]
