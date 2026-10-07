"""chunking.textual: fixed / recursive / header. 구간이 평문의 리터럴 슬라이스이고 공백 아닌 글자를 빠짐없이 덮는지, LangChain과 같은 경계인지."""
import unittest

import numpy as np

import support
from support import FIX
from chunking import base, chunk_document
from chunking import textual as T
from chunking.io import coverage_ok
from chunking.v1 import splitters as old
from common.docjson import load_doc


def ctx_for(key, cpt=4.0):
    return base.DocContext(load_doc(FIX / f"example_{key}.json"), chars_per_token=cpt)


def check_spans(tc, ctx, spans):
    tc.assertTrue(spans)
    prev = 0
    for s, e in spans:
        tc.assertTrue(0 <= s < e <= len(ctx.text) and s >= prev, (s, e, prev))
        prev = e
    tc.assertTrue(coverage_ok(ctx.text, spans), "공백이 아닌 글자가 어느 청크에도 없다")


class TestFixed(unittest.TestCase):
    def test_same_as_v1_fixed(self):
        for k in "ac":
            ctx = ctx_for(k)
            self.assertEqual(chunk_document("fixed", ctx, size_chars=160), old.fixed_spans(ctx.text, 160))
            check_spans(self, ctx, chunk_document("fixed", ctx, size_chars=160))

    def test_size_from_tokens(self):
        ctx = ctx_for("a", cpt=4.0)
        self.assertEqual(chunk_document("fixed", ctx, T=40), chunk_document("fixed", ctx, size_chars=160))

    def test_needs_size(self):
        with self.assertRaises(ValueError):
            chunk_document("fixed", ctx_for("a"))


class TestRecursive(unittest.TestCase):
    def test_cap_and_coverage(self):
        for k in "ac":
            ctx = ctx_for(k)
            for size in (60, 160, 400):
                spans = chunk_document("recursive", ctx, size_chars=size)
                check_spans(self, ctx, spans)
                self.assertLessEqual(max(e - s for s, e in spans), size)

    def test_prefers_paragraph_boundaries(self):
        ctx = ctx_for("a")
        spans = chunk_document("recursive", ctx, size_chars=400)
        starts = {s for s, _ in spans}
        paragraph_starts = {p.start for p in ctx.pieces}
        self.assertTrue(len(starts & paragraph_starts) >= len(spans) // 2)


class TestHeader(unittest.TestCase):
    def test_splits_at_headings(self):
        ctx = ctx_for("a")
        spans = chunk_document("header", ctx)
        check_spans(self, ctx, spans)
        starts = {s for s, _ in spans}
        for s in ctx.head_level:                       # 모든 제목(H1~H3)이 새 조각을 시작한다
            self.assertIn(s, starts)

    def test_matches_langchain_char_stream(self):
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        clean = lambda t: "".join(ch for ch in t if ch.isprintable() and not ch.isspace())
        for k in "ac":
            ctx = ctx_for(k)
            mine = [clean(ctx.text[s:e]) for s, e in chunk_document("header", ctx)]
            lc = [clean(d.page_content) for d in MarkdownHeaderTextSplitter(
                [("#", "H1"), ("##", "H2"), ("###", "H3")], strip_headers=False).split_text(ctx.text)]
            self.assertEqual(mine, [x for x in lc if x])

    def test_resplit_cap(self):
        ctx = ctx_for("a")
        pure = chunk_document("header", ctx)
        cap = 120
        spans = chunk_document("header", ctx, size_chars=cap)
        check_spans(self, ctx, spans)
        self.assertLessEqual(max(e - s for s, e in spans), cap)
        self.assertGreater(len(spans), len(pure))
        short = [(s, e) for s, e in pure if e - s <= cap]
        for seg in short:                                # 상한 이하인 조각은 그대로 남는다
            self.assertIn(seg, spans)

    def test_corrupt_langchain_output_raises(self):
        """LangChain 출력의 글자 열이 원문과 어긋나면 조용히 넘어가지 않고 오류를 낸다."""
        from types import SimpleNamespace
        from unittest import mock

        class Fake:
            def __init__(self, *a, **k):
                pass

            def split_text(self, text):
                return [SimpleNamespace(page_content="완전히 다른 글")]

        with mock.patch.object(T, "MarkdownHeaderTextSplitter", Fake):
            with self.assertRaises(ValueError):
                T.header_spans("# a\n\nb c d", 3, None)


if __name__ == "__main__":
    unittest.main()
