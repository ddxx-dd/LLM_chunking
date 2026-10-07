"""chunking.semantic / smart_a: 임계값 의미 분할과 H1 섹션 강제 경계."""
import unittest

import numpy as np

import support
from support import FIX, runtime_for
from chunking import base, chunk_document
from chunking.embed import Runtime
from chunking.semantic import adjacent_distances, semantic_groups
from chunking.smart_a import section_level, section_starts
from common.docjson import load_doc


def angles(deg):
    a = np.deg2rad(deg)
    return np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)


def is_partition(groups, n):
    return bool(groups) and groups[0][0] == 0 and groups[-1][1] == n and all(a[1] == b[0] for a, b in zip(groups, groups[1:])) and all(a < b for a, b in groups)


class TestSemanticGroups(unittest.TestCase):
    E = angles([0, 5, 10, 90, 95, 100, 200, 205])

    def test_cuts_at_large_distances(self):
        self.assertEqual(semantic_groups(self.E, 70), [(0, 3), (3, 6), (6, 8)])

    def test_forced_cut_added(self):
        self.assertEqual(semantic_groups(self.E, 70, forced=[0]), [(0, 1), (1, 3), (3, 6), (6, 8)])

    def test_higher_percentile_means_fewer_cuts(self):
        counts = [len(semantic_groups(self.E, p)) for p in (10, 50, 70, 95)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_partition_and_edge_cases(self):
        for n in (0, 1, 2, 9):
            rng = np.random.default_rng(n)
            E = rng.normal(size=(n, 4)).astype(np.float32)
            if n:
                E /= np.linalg.norm(E, axis=1, keepdims=True)
            g = semantic_groups(E, 80)
            if n:
                self.assertTrue(is_partition(g, n))
            else:
                self.assertEqual(g, [])

    def test_distances(self):
        d = adjacent_distances(angles([0, 90, 90]))
        np.testing.assert_allclose(d, [1.0, 0.0], atol=1e-6)


class TestSemanticChunker(unittest.TestCase):
    def test_fixture_partition(self):
        for k in "ac":
            ctx = base.DocContext(load_doc(FIX / f"example_{k}.json"), runtime_for(k.upper()))
            spans = chunk_document("semantic", ctx, p=70)
            self.assertEqual(spans[0][0], base.whole_doc(ctx.text)[0][0])
            self.assertEqual(spans[-1][1], base.whole_doc(ctx.text)[0][1])
            for (s, e), (s2, _) in zip(spans, spans[1:]):
                self.assertLessEqual(e, s2)

    def test_single_unit_doc(self):
        doc = {"doc_id": "x", "dataset": "docx_corpus", "fmt": "docx", "lang": "en",
               "blocks": [{"id": "b", "type": "paragraph", "text": "Only one sentence here."}]}
        ctx = base.DocContext(doc, runtime_for("C"))
        self.assertEqual(chunk_document("semantic", ctx, p=90), [(0, len(ctx.text))])


class TestSectionStructure(unittest.TestCase):
    def test_section_level(self):
        self.assertEqual(section_level({0: 1, 10: 2, 20: 2}), 2)        # # 하나뿐이면 ## 수준
        self.assertEqual(section_level({0: 1, 10: 1, 20: 2}), 1)
        self.assertIsNone(section_level({0: 1}))
        self.assertIsNone(section_level({}))

    def test_section_starts_exclude_first_piece(self):
        doc = load_doc(FIX / "example_a.json")
        ctx = base.DocContext(doc)
        starts = section_starts(ctx)
        self.assertEqual(len(starts), 2)                                 # ## 두 개 중 문서 맨 앞 조각이 아닌 것들
        for s in starts:
            self.assertIn(s, ctx.head_level)


def two_section_doc():
    return {"doc_id": "two", "dataset": "docx_corpus", "fmt": "docx", "lang": "en", "blocks": [
        {"id": "a0", "type": "heading", "level": 1, "text": "Alpha"},
        {"id": "a1", "type": "paragraph", "text": "alpha one. alpha two. alpha three."},
        {"id": "b0", "type": "heading", "level": 1, "text": "Beta"},
        {"id": "b1", "type": "paragraph", "text": "beta one. beta two. beta three."}]}


class TestA(unittest.TestCase):
    def same_embedding_runtime(self):
        # 모든 문장이 같은 방향 -> 의미 거리가 전부 같아서 semantic은 어디서도 자르지 않는다
        return Runtime(lambda ts: np.tile(np.array([[1.0, 0.0]], np.float32), (len(ts), 1)), lambda ts: [len(t.split()) for t in ts])

    def test_forces_section_boundary(self):
        ctx = base.DocContext(two_section_doc(), self.same_embedding_runtime())
        self.assertEqual(len(chunk_document("semantic", ctx, p=90)), 1)
        spans = chunk_document("A", ctx, p=90)
        self.assertEqual(len(spans), 2)
        self.assertTrue(ctx.text[spans[1][0]:].startswith("# Beta"))     # 섹션 경계가 정확히 제목 앞

    def test_no_forced_without_structure(self):
        doc = {"doc_id": "n", "dataset": "docx_corpus", "fmt": "docx", "lang": "en",
               "blocks": [{"id": "p", "type": "paragraph", "text": "one. two. three. four."}]}
        ctx = base.DocContext(doc, self.same_embedding_runtime())
        self.assertEqual(chunk_document("A", ctx, p=90), chunk_document("semantic", ctx, p=90))

    def test_fixture_a_chunk_starts_at_h2(self):
        ctx = base.DocContext(load_doc(FIX / "example_a.json"), runtime_for("A"))
        spans = chunk_document("A", ctx, p=90)
        starts = {s for s, _ in spans}
        for s in section_starts(ctx):                                    # 모든 섹션 시작이 청크 시작
            self.assertIn(s, starts)


if __name__ == "__main__":
    unittest.main()
