"""크기 제어 순수 DP(chunking/size_dp.partition) 테스트 - 텍스트·임베딩 없이 숫자만. 전수 탐색과 점수가 같은지가 핵심이다."""
import unittest

import numpy as np

import support  # noqa: F401  (src 경로 설정)
from chunking.sizectl.size_dp import partition


def sizer(sizes):
    pre = [0]
    for s in sizes:
        pre.append(pre[-1] + s)
    return lambda a, b: pre[b] - pre[a]


def brute(sizes, gains, mn, mx):
    """모든 분할을 열거해 사전식 최적 (부족분 합 최소, 이득 최대)을 구한다."""
    n, size, best = len(sizes), sizer(sizes), None
    for mask in range(2 ** (n - 1)):
        cuts = [i for i in range(n - 1) if mask >> i & 1]
        b = [0] + [c + 1 for c in cuts] + [n]
        segs = list(zip(b, b[1:]))
        if any(size(a, e) > mx and e - a > 1 for a, e in segs):
            continue
        key = (-sum(max(0, mn - size(a, e)) for a, e in segs), sum(gains[c] for c in cuts))
        if best is None or key[0] > best[0][0] or (key[0] == best[0][0] and key[1] > best[0][1] + 1e-12):
            best = (key, segs)
    return best


class Partition(unittest.TestCase):
    def test_worked_example_from_the_plan(self):
        sizes = [60, 50, 120, 40, 90, 80, 70]
        d = [0.10, 0.70, 0.20, 0.60, 0.15, 0.50]
        gains = [x - 0.30 for x in d]
        groups, (short, gain) = partition(7, sizer(sizes), gains, 100, 200)
        self.assertEqual(groups, [(0, 2), (2, 3), (3, 5), (5, 7)])
        self.assertEqual([sizer(sizes)(a, b) for a, b in groups], [110, 120, 130, 150])
        self.assertEqual(short, 0)
        self.assertAlmostEqual(gain, 0.15)
        cuts = [i for i, g in enumerate(gains) if g > 0]                       # 임계값 컷만 쓰면 마지막이 70토큰
        b = [0] + [c + 1 for c in cuts] + [7]
        self.assertEqual([sizer(sizes)(x, y) for x, y in zip(b, b[1:])], [110, 160, 170, 70])

    def test_score_equals_brute_force_on_random_small_inputs(self):
        rng = np.random.default_rng(11)
        for _ in range(60):
            n = int(rng.integers(2, 9))
            sizes = [int(x) for x in rng.integers(5, 90, n)]
            gains = [float(x) for x in rng.normal(0, 0.4, n - 1)]
            mn, mx = int(rng.integers(20, 60)), int(rng.integers(90, 170))
            groups, (short, gain) = partition(n, sizer(sizes), gains, mn, mx)
            (bshort, bgain), _segs = brute(sizes, gains, mn, mx)
            self.assertEqual(-short, bshort, (sizes, gains, mn, mx))
            self.assertAlmostEqual(gain, bgain, places=9, msg=(sizes, gains, mn, mx))
            self.assertEqual((groups[0][0], groups[-1][1]), (0, n))
            self.assertTrue(all(x[1] == y[0] for x, y in zip(groups, groups[1:])))

    def test_without_binding_constraints_it_cuts_exactly_where_gain_is_positive(self):
        gains = [-0.4, 0.4, -0.1, -0.2, 0.3, -0.5, -0.1]
        groups, _ = partition(8, sizer([10] * 8), gains, 1, 10_000)
        self.assertEqual(groups, [(0, 2), (2, 5), (5, 8)])

    def test_max_is_hard_but_a_single_unit_over_max_is_kept(self):
        sizes = [5, 300, 5, 40, 40]
        groups, _ = partition(5, sizer(sizes), [-1.0] * 4, 10, 100)
        self.assertIn((1, 2), groups)                                          # 300토큰 단위는 그대로 한 그룹
        self.assertTrue(all(sizer(sizes)(a, b) <= 100 for a, b in groups if b - a > 1))

    def test_min_beats_a_huge_gain_when_a_feasible_partition_exists(self):
        # 경계 0의 gain이 아주 커도(forced 보정) 자르면 3토큰짜리가 생기므로 min이 이긴다
        groups, (short, _) = partition(3, sizer([3, 30, 30]), [1000.0, -0.5], 10, 100)
        self.assertEqual((groups, short), ([(0, 3)], 0))
        groups, _ = partition(3, sizer([20, 30, 30]), [1000.0, -0.5], 10, 100)  # 잘라도 min 이상이면 큰 gain을 따른다
        self.assertEqual(groups, [(0, 1), (1, 3)])

    def test_unavoidable_shortfall_is_reported(self):
        groups, (short, _) = partition(2, sizer([40, 300]), [0.2], 100, 200)    # 표(300) 옆의 40토큰은 합칠 수 없다
        self.assertEqual((groups, short), ([(0, 1), (1, 2)], 60))

    def test_empty_and_single(self):
        self.assertEqual(partition(0, sizer([]), [], 10, 20), ([], (0, 0.0)))
        self.assertEqual(partition(1, sizer([5]), [], 10, 20), ([(0, 1)], (5, 0.0)))

    def test_deterministic(self):
        sizes, gains = [30, 30, 30, 30, 30], [0.1, 0.1, 0.1, 0.1]
        self.assertEqual(partition(5, sizer(sizes), gains, 50, 100), partition(5, sizer(sizes), gains, 50, 100))


if __name__ == "__main__":
    unittest.main()
