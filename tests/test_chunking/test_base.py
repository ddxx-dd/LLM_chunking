"""chunking.base / embed: 구간 도구, 문서 컨텍스트, 등록부, 단위 캐시."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import support
from support import FIX, runtime_for
import chunking
from chunking import base
from chunking.embed import Runtime
from chunking.sizectl.semantic_core import sentence_spans as old_sentence_spans
from common.docjson import load_doc, render


class TestSpans(unittest.TestCase):
    def test_sentence_spans_same_as_engine(self):
        """새 sentence_spans는 srt·옛 엔진이 쓰는 sizectl 것과 같은 결과여야 한다(같은 문장 나누기)."""
        texts = [render(load_doc(FIX / f"example_{k}.json"))[0] for k in "abc"]
        texts += ["", "한 문장. 두 문장!\n\n세 문장? 네 번째", "no break", "  앞뒤 공백.  다음.  "]
        for t in texts:
            self.assertEqual(base.sentence_spans(t), old_sentence_spans(t), t[:30])
        t = texts[0]
        self.assertEqual(base.sentence_spans(t, 10, 80), old_sentence_spans(t, 10, 80))

    def test_whole_doc_trims(self):
        self.assertEqual(base.whole_doc("  ab c \n"), [(2, 6)])
        self.assertEqual(base.whole_doc("   "), [])

    def test_groups_and_spans(self):
        self.assertEqual(base.groups_from_cuts(5, [1, 3]), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(base.groups_from_cuts(3, []), [(0, 3)])
        units = [(0, 4), (5, 9), (10, 14)]
        self.assertEqual(base.spans_from_groups(units, [(0, 2), (2, 3)]), [(0, 9), (10, 14)])


class TestContext(unittest.TestCase):
    def setUp(self):
        self.doc = load_doc(FIX / "example_a.json")

    def test_text_and_levels(self):
        ctx = base.DocContext(self.doc)
        self.assertEqual(ctx.text, render(self.doc)[0])
        self.assertEqual(sorted(ctx.head_level.values()), [1, 2, 2])           # 예시 A: # 1개, ## 2개
        for s in ctx.head_level:
            self.assertTrue(ctx.text[s:].startswith("#"))

    def test_size_chars(self):
        ctx = base.DocContext(self.doc, chars_per_token=3.4)
        self.assertEqual(ctx.size_chars(T=100), 340)
        self.assertEqual(ctx.size_chars(T=100, size_chars=500), 500)             # 글자수가 우선
        self.assertIsNone(ctx.size_chars())
        with self.assertRaises(ValueError):
            base.DocContext(self.doc).size_chars(T=100)

    def test_units_need_runtime(self):
        with self.assertRaises(RuntimeError):
            base.DocContext(self.doc).units("sent")

    def test_units_shapes(self):
        ctx = base.DocContext(self.doc, runtime_for("A"))
        for kind in ("sent", "sentg"):
            u = ctx.units(kind)
            self.assertEqual(len(u.spans), len(u.emb))
            self.assertEqual(len(u.spans), len(u.ntok))
            np.testing.assert_allclose(np.linalg.norm(u.emb, axis=1), 1.0, atol=1e-5)
            self.assertIs(ctx.units(kind), u)                                      # 한 번만 만든다
        self.assertEqual(ctx.levels(ctx.units("sentg").spans).tolist().count(0) + 3, len(ctx.units("sentg")))


class TestRegistry(unittest.TestCase):
    def test_seven_chunkers(self):
        self.assertEqual(set(chunking.CHUNKERS), {"fixed", "recursive", "header", "semantic", "A", "B", "E"})
        for name, ch in chunking.REGISTRY.items():
            self.assertEqual(ch.name, name)
            self.assertIn(ch.knob, ("p", "T"))

    def test_unknown_name(self):
        with self.assertRaises(KeyError):
            chunking.chunk_document("nope", base.DocContext(load_doc(FIX / "example_a.json")))

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            @base.register
            class Dup(base.Chunker):
                name = "A"


class TestUnitCache(unittest.TestCase):
    def test_roundtrip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Runtime(lambda ts: np.ones((len(ts), 4), np.float32) / 2, lambda ts: [1] * len(ts), cache_dir=d)
            spans = [(0, 3), (4, 8)]
            self.assertIsNone(rt.cache_get("ds", "doc/1", "sent", spans))
            rt.cache_put("ds", "doc/1", "sent", spans, np.ones((2, 4), np.float32) / 2, [3, 4])
            emb, ntok = rt.cache_get("ds", "doc/1", "sent", spans)
            self.assertEqual(emb.shape, (2, 4))
            self.assertEqual(ntok.tolist(), [3, 4])
            self.assertIsNone(rt.cache_get("ds", "doc/1", "sent", [(0, 3), (4, 9)]))     # 구간이 다르면 쓰지 않는다
            self.assertTrue(any(Path(d).rglob("*.npz")))

    def test_context_uses_cache(self):
        doc = load_doc(FIX / "example_a.json")
        calls = []
        with tempfile.TemporaryDirectory() as d:
            def enc(ts):
                calls.append(len(ts))
                return np.ones((len(ts), 3), np.float32) / np.sqrt(3)
            rt = Runtime(enc, lambda ts: [len(t.split()) for t in ts], cache_dir=d)
            base.DocContext(doc, rt).units("sent")
            base.DocContext(doc, rt).units("sent")                                       # 두 번째는 캐시
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
