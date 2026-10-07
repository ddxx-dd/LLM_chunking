"""chunking.smart_e (E 중심 병합): 단순 구현과 같은 결과, 사슬 병합이 늦게 끊김, 제목 경계 감점."""
import unittest

import numpy as np

import support
from support import FIX, runtime_for
from chunking import base, chunk_document
from chunking import smart_e as E
from common.docjson import load_doc


def angles(deg):
    a = np.deg2rad(deg)
    return np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)


def naive_merge(emb, own, lv, theta, delta, decay=0.5):
    """매 단계 전체를 다시 계산하는 단순 구현(O(n^3)). 동률은 왼쪽 쌍."""
    groups = [[i, i + 1] for i in range(len(emb))]
    vs = [emb[i].astype(float) * own[i] for i in range(len(emb))]
    while len(groups) > 1:
        best = None
        for k in range(len(groups) - 1):
            cl, cr = vs[k] / np.linalg.norm(vs[k]), vs[k + 1] / np.linalg.norm(vs[k + 1])
            r = groups[k + 1][0]
            eff = float(cl @ cr) - (delta * decay ** (lv[r] - 1) if lv[r] > 0 else 0.0)
            if best is None or eff > best[0] + 1e-12:
                best = (eff, k)
        if best[0] < theta:
            break
        k = best[1]
        groups[k] = [groups[k][0], groups[k + 1][1]]
        vs[k] = vs[k] + vs[k + 1]
        del groups[k + 1], vs[k + 1]
    return [tuple(g) for g in groups]


class TestMerge(unittest.TestCase):
    def test_equals_naive(self):
        rng = np.random.default_rng(2)
        for _ in range(60):
            n = int(rng.integers(2, 14))
            emb = rng.normal(size=(n, 6))
            emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
            own = rng.integers(5, 80, n)
            lv = rng.integers(0, 3, n)
            theta, delta, decay = float(rng.random() * 0.8 - 0.2), float(rng.random() * 0.4), float(rng.choice([0.25, 0.5, 1.0]))
            got = E.merge_blocks(emb, own, lv, theta, delta, decay)
            self.assertEqual(got, naive_merge(emb, own, lv, theta, delta, decay))

    def test_chain_ends_late_with_centroids(self):
        """각도 0,15,30,45,60: 인접 코사인은 모두 .966이지만 중심으로 비교하면 theta=.8에서 사슬이 끊긴다."""
        emb = angles([0, 15, 30, 45, 60])
        own, lv = np.ones(5), np.zeros(5, dtype=int)
        self.assertEqual(E.merge_blocks(emb, own, lv, 0.8, 0.0), [(0, 2), (2, 5)])
        self.assertEqual(E.merge_blocks(emb, own, lv, 0.7, 0.0), [(0, 5)])
        single_link_would_merge_all = all(float(emb[i] @ emb[i + 1]) > 0.8 for i in range(4))
        self.assertTrue(single_link_would_merge_all)

    def test_heading_penalty_blocks_merge(self):
        emb = angles([0, 10, 20, 30])
        own, lv = np.ones(4), np.array([0, 0, 1, 0])
        self.assertEqual(E.merge_blocks(emb, own, lv, 0.9, 0.0), [(0, 4)])
        g = E.merge_blocks(emb, own, lv, 0.9, 0.3)
        self.assertIn((0, 2), g)
        self.assertNotIn(1, [a for a, _ in g])

    def test_level2_penalty_is_smaller(self):
        emb = angles([0, 10, 20, 30])
        own = np.ones(4)
        sim = float(emb[1] @ emb[2])                 # 경계 1|2의 초기 코사인
        delta = 0.2
        theta = sim - delta * 0.75                    # 수준1 감점(0.2)이면 못 넘고, 수준2 감점(0.1)이면 넘는 값
        g1 = E.merge_blocks(emb, own, np.array([0, 0, 1, 0]), theta, delta)
        g2 = E.merge_blocks(emb, own, np.array([0, 0, 2, 0]), theta, delta)
        self.assertEqual(g2, [(0, 4)])
        self.assertNotEqual(g1, [(0, 4)])

    def test_trivial_inputs(self):
        self.assertEqual(E.merge_blocks(np.zeros((0, 2), np.float32), [], [], 0.5, 0.1), [])
        self.assertEqual(E.merge_blocks(angles([0]), np.ones(1), np.zeros(1), 0.5, 0.1), [(0, 1)])


class TestGroups(unittest.TestCase):
    def test_partition(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            n = int(rng.integers(1, 60))
            emb = rng.normal(size=(n, 16))
            emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
            own, lv = rng.integers(3, 60, n), rng.integers(0, 3, n)
            for p in (75, 90, 98):
                g = E.e_groups(emb, own, lv, p)
                self.assertEqual(g[0][0], 0)
                self.assertEqual(g[-1][1], n)
                self.assertTrue(all(a[1] == b[0] for a, b in zip(g, g[1:])))

    def test_higher_p_means_fewer_chunks(self):
        rng = np.random.default_rng(4)
        emb = rng.normal(size=(80, 8))
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        counts = [len(E.e_groups(emb, np.ones(80), np.zeros(80, dtype=int), p)) for p in (50, 75, 90, 98)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_delta_default_is_sigma_multiple(self):
        emb = angles([0, 10, 20, 30, 40, 50])
        sims = (emb[:-1] * emb[1:]).sum(1)
        own, lv = np.ones(6), np.array([0, 0, 0, 1, 0, 0])
        a = E.e_groups(emb, own, lv, 70, delta=None, delta_sigma=2.0)
        b = E.e_groups(emb, own, lv, 70, delta=2.0 * float(sims.std()))
        self.assertEqual(a, b)


class TestChunker(unittest.TestCase):
    def test_fixture_partition(self):
        for k in "ac":
            ctx = base.DocContext(load_doc(FIX / f"example_{k}.json"), runtime_for(k.upper()))
            spans = chunk_document("E", ctx, p=70)
            self.assertTrue(spans)
            for (s, e), (s2, _) in zip(spans, spans[1:]):
                self.assertLessEqual(e, s2)


if __name__ == "__main__":
    unittest.main()
