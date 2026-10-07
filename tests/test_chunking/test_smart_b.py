"""chunking.smart_b (B 응집도 DP): 식이 정의대로인지, DP가 전수 탐색과 같은 최적인지, 제목 보너스가 의도대로 작동하는지."""
import unittest

import numpy as np

import support
from support import FIX, runtime_for
from chunking import base, chunk_document
from chunking import smart_b as B
from common.docjson import load_doc


def unit_vectors(rng, n, d=6):
    e = rng.normal(size=(n, d))
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)


def angles(deg):
    a = np.deg2rad(deg)
    return np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)


def total_cost(emb, groups, bonus, lam):
    return sum((b - a) - float(((emb[a:b].astype(float).sum(0)) ** 2).sum()) / (b - a) + lam - (bonus[a] if a > 0 else 0.0) for a, b in groups)


class TestFormulas(unittest.TestCase):
    def test_cohesion_equals_sum_of_squared_deviations(self):
        """c(a,b) = m - ||S||^2/m  ==  sum ||e - mean||^2 (정규화 벡터)."""
        E = np.random.default_rng(0).normal(size=(7, 6))
        E /= np.linalg.norm(E, axis=1, keepdims=True)                 # float64로 정확히 길이 1
        m = len(E)
        c = m - float((E.sum(0) ** 2).sum()) / m
        dev = float(((E - E.mean(0)) ** 2).sum())
        self.assertAlmostEqual(c, dev, places=9)

    def test_local_deltas_matches_naive(self):
        E = unit_vectors(np.random.default_rng(1), 40)
        got = B.local_deltas(E, 5)
        for i in range(len(E) - 1):
            L, R = E[max(0, i - 4):i + 1].astype(float), E[i + 1:i + 6].astype(float)
            mL, mR = len(L), len(R)
            want = mL * mR / (mL + mR) * float(np.sum((L.mean(0) - R.mean(0)) ** 2))
            self.assertAlmostEqual(got[i], want, places=9)

    def test_delta_is_cost_reduction_of_cutting(self):
        """Delta_i = (두 창을 합친 응집도 비용) - (각각의 응집도 비용 합)  (Ward 공식)."""
        E = unit_vectors(np.random.default_rng(2), 12).astype(float)
        cost = lambda x: len(x) - float((x.sum(0) ** 2).sum()) / len(x)
        i, w = 5, 4
        L, R = E[i + 1 - w:i + 1], E[i + 1:i + 1 + w]
        want = cost(np.vstack([L, R])) - cost(L) - cost(R)
        self.assertAlmostEqual(float(B.local_deltas(E.astype(np.float32), w)[i]), want, places=5)

    def test_bonus_array(self):
        b = B.bonus_array([0, 1, 2, 3], lam=2.0, beta=1.0, decay=0.5)
        np.testing.assert_allclose(b, [0.0, 2.0, 1.0, 0.5])
        np.testing.assert_allclose(B.bonus_array([1, 1], 3.0, 0.0, 0.5), [0.0, 0.0])


class TestDP(unittest.TestCase):
    def test_equals_exhaustive_search(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            n = int(rng.integers(2, 9))
            E = unit_vectors(rng, n)
            own = rng.integers(5, 70, n).astype(float)
            lv = rng.integers(0, 3, n)
            lam = float(rng.random() * 2)
            cap = int(rng.integers(60, 200))
            bonus = B.bonus_array(lv, lam, 1.0, 0.5)
            got = B.coherence_dp(E, own, bonus, lam, cap)
            pt = np.concatenate([[0], np.cumsum(own)])
            best = None
            for mask in range(2 ** (n - 1)):
                cuts = [i for i in range(n - 1) if mask >> i & 1]
                b = [0] + [c + 1 for c in cuts] + [n]
                segs = list(zip(b, b[1:]))
                if any(pt[e] - pt[a] > cap and e - a > 1 for a, e in segs):
                    continue
                c = total_cost(E, segs, bonus, lam)
                best = c if best is None or c < best else best
            self.assertAlmostEqual(total_cost(E, got, bonus, lam), best, places=6)
            self.assertEqual(got[0][0], 0)
            self.assertEqual(got[-1][1], n)

    def test_two_topics(self):
        E = angles([0, 5, 10, 100, 105, 110])
        g = B.b_groups(E, np.ones(6), np.zeros(6), p=70)
        self.assertEqual(g, [(0, 3), (3, 6)])

    def test_stray_sentence_absorbed(self):
        """한 주제 안에 엉뚱한 문장 하나가 끼어도(양옆 거리가 모두 큼) 조각내지 않는다."""
        E = angles([0, 5, 10, 60, 15, 20, 100, 105, 110])
        g = B.b_groups(E, np.ones(9), np.zeros(9), p=70)
        self.assertEqual(g, [(0, 6), (6, 9)])

    def test_cap_limits_candidate_length(self):
        E = angles([0, 1, 2, 3, 4, 5])
        g = B.coherence_dp(E, np.full(6, 10.0), np.zeros(6), lam=5.0, cap=20)       # 한 청크 후보는 20토큰(문장 2개)까지
        self.assertTrue(all(b - a <= 2 for a, b in g))

    def test_single_big_unit_allowed(self):
        E = angles([0, 90])
        g = B.coherence_dp(E, np.array([100.0, 5.0]), np.zeros(2), lam=0.0, cap=10)
        self.assertEqual(g[0], (0, 1))


class TestHeadingBonus(unittest.TestCase):
    E = angles([0, 3, 6, 9, 12, 15])               # 완만하게 변하는 6문장
    LV = np.array([0, 0, 0, 1, 0, 0])              # 4번째 문장이 H1로 시작

    def groups(self, beta, decay=0.5, lv=None):
        lam = 0.5
        return B.coherence_dp(self.E, np.ones(6), B.bonus_array(self.LV if lv is None else lv, lam, beta, decay), lam)

    def test_no_bonus_no_cut(self):
        self.assertEqual(self.groups(beta=0.0), [(0, 6)])

    def test_bonus_makes_heading_cut(self):
        self.assertEqual(self.groups(beta=3.0), [(0, 3), (3, 6)])

    def test_level2_gets_half_bonus(self):
        lv2 = np.array([0, 0, 0, 2, 0, 0])
        self.assertEqual(self.groups(beta=3.0, decay=0.5, lv=lv2), [(0, 3), (3, 6)])      # 0.5*3*0.5*... 이어도 임계를 넘는 값
        self.assertEqual(self.groups(beta=1.0, decay=0.01, lv=lv2), [(0, 6)])             # 감쇠가 크면(값이 작으면) 효과가 없다


class TestChunker(unittest.TestCase):
    def test_fixture_partition(self):
        for k in "ac":
            ctx = base.DocContext(load_doc(FIX / f"example_{k}.json"), runtime_for(k.upper()))
            spans = chunk_document("B", ctx, p=70)
            self.assertTrue(spans)
            for (s, e), (s2, _) in zip(spans, spans[1:]):
                self.assertLessEqual(e, s2)

    def test_higher_p_fewer_chunks(self):
        ctx = base.DocContext(load_doc(FIX / "example_a.json"), runtime_for("A"))
        counts = [len(chunk_document("B", ctx, p=p)) for p in (30, 70, 98)]
        self.assertEqual(counts, sorted(counts, reverse=True))


if __name__ == "__main__":
    unittest.main()
