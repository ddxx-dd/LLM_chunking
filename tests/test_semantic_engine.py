"""의미 분할 공유 엔진(semantic_core) 단위 테스트 - 가짜 임베딩(키워드 축)과 공백 단위 토큰 카운터.
핵심은 LangChain SemanticChunker와 같은 경계를 만드는지(동치 테스트)와 크기 정리 규칙이다."""
import unittest
import zlib

import numpy as np


from support import FIX, AXES, FILES, TOPIC_AXES, make_stub, count_words, words
from langchain_experimental.text_splitter import SemanticChunker

from chunking.semantic_core import (SENTENCE_SPLIT_REGEX, SemanticEngine, combined_texts,
                                    consecutive_distances, sentence_spans)
from common.docjson import load_doc, to_text



def joined(text, units, a, b):
    return " ".join(text[s:e] for s, e in units[a:b])


class SentenceSpans(unittest.TestCase):
    def test_offsets_are_literal_and_trimmed(self):
        t = "One. Two?\n\n  Three! four\n"
        spans = sentence_spans(t)
        self.assertEqual([t[s:e] for s, e in spans], ["One.", "Two?", "Three!", "four"])

    def test_sub_range_uses_original_offsets(self):
        t = "skip. Alpha one. Alpha two.\n\nlater"
        s0 = t.index("Alpha")
        e0 = t.index("\n\nlater")
        spans = sentence_spans(t, s0, e0)
        self.assertEqual([t[s:e] for s, e in spans], ["Alpha one.", "Alpha two."])

    def test_empty_and_blank(self):
        self.assertEqual(sentence_spans(""), [])
        self.assertEqual(sentence_spans(" \n\n "), [])


class Distances(unittest.TestCase):
    def test_combined_texts_follow_langchain(self):
        self.assertEqual(combined_texts(["a", "b", "c"], 1), ["a b", "a b c", "b c"])
        self.assertEqual(combined_texts(["a", "b", "c"], 0), ["a", "b", "c"])

    def test_distance_is_zero_for_same_topic_and_large_across_topics(self):
        emb = make_stub(TOPIC_AXES)._client
        d = consecutive_distances(emb, [words("alpha", 5), words("alpha", 7), words("beta", 6), words("beta", 6)], buffer_size=0)
        self.assertLess(d[0], 1e-3)
        self.assertGreater(d[1], 0.5)
        self.assertLess(d[2], 1e-3)


def jittered(axes):
    """키워드 축 가짜 임베딩에 글 내용으로 정해지는 아주 작은 변동(1e-3)을 더한다. 거리가 정확히 같은(동점) 경계가 없어져서
    LangChain(sklearn)과 우리 계산의 1e-16 수준 차이로 컷이 갈리는 일이 없다. 실제 임베딩에서는 동점이 사실상 생기지 않는다."""
    base = make_stub(axes)

    def vec(t):
        v = np.asarray(base._client.encode([t])[0], float)
        noise = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).normal(size=len(v))
        v = v + 1e-3 * noise
        return v / np.linalg.norm(v)

    class Emb:
        def embed_documents(self, ts):
            return [vec(t).tolist() for t in ts]

        def embed_query(self, t):
            return vec(t).tolist()

        @property
        def _client(self):
            class Client:
                def encode(s, ts, normalize_embeddings=True):
                    return np.array([vec(t) for t in ts])
            return Client()
    return Emb()


class EquivalenceWithLangChain(unittest.TestCase):
    """크기 정리 전의 경계가 LangChain SemanticChunker가 정한 경계와 같다 -> 기준선을 제대로 재현한다는 증거.
    다른 점 하나: LangChain은 텍스트 양끝의 줄바꿈이 만드는 빈 문자열도 "문장"으로 세므로, 비교는 양끝 공백을 뺀 텍스트로 한다
    (우리 엔진은 빈 조각을 버린다)."""

    def check(self, text, axes, pct, buffer_size):
        text = text.strip()
        stub = jittered(axes)
        lc = SemanticChunker(stub, buffer_size=buffer_size, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=pct,
                             sentence_split_regex=SENTENCE_SPLIT_REGEX).split_text(text)
        units = sentence_spans(text)
        eng = SemanticEngine(stub._client, count_words, percentile=pct, buffer_size=buffer_size)
        groups, _d, _tau = eng.breakpoint_groups(text, units)
        ours = [" ".join(joined(text, units, a, b).split()) for a, b in groups]
        self.assertEqual(ours, [" ".join(c.split()) for c in lc])

    def test_fixture_documents(self):
        for key in "ABC":
            text = to_text(load_doc(FIX / f"{FILES[key]}.json"))
            for pct in (60, 85, 90):
                for buf in (0, 1):                                            # 0 = 우리 기본, 1 = LangChain 기본
                    with self.subTest(example=key, percentile=pct, buffer_size=buf):
                        self.check(text, AXES[key], pct, buf)

    def test_synthetic_topics(self):
        # 문장 길이를 모두 달리해서 거리가 정확히 같은(동점) 경계가 없게 한다 - 동점은 부동소수점 잡음으로 갈리기 때문
        text = "\n\n".join([words("alpha", 8 + i) for i in range(4)] + [words("beta", 9 + i) for i in range(3)]
                          + [words("gamma", 10 + i) for i in range(4)])
        for pct in (50, 80, 95):
            for buf in (0, 1, 2):
                with self.subTest(percentile=pct, buffer_size=buf):
                    self.check(text, TOPIC_AXES, pct, buf)


class SizeRegularization(unittest.TestCase):
    def engine(self, mn, mx, pct=90):
        return SemanticEngine(make_stub(TOPIC_AXES)._client, count_words, percentile=pct, min_tokens=mn, max_tokens=mx)

    def units(self, parts):
        text = "\n\n".join(parts)
        return text, sentence_spans(text)

    def test_forced_boundary_is_cut_even_when_meaning_does_not_change(self):
        text, units = self.units([words("alpha", 10)] * 6)
        eng = self.engine(mn=5, mx=200, pct=100)                       # percentile 100 -> 의미 컷 없음
        self.assertEqual(eng.chunk_spans(text, units, forced=[2]), [(units[0][0], units[2][1]), (units[3][0], units[5][1])])
        self.assertEqual(eng.stats["forced_cuts"], 1)
        self.assertEqual(eng.stats["threshold_cuts"], 0)

    def test_oversized_group_is_cut_at_the_topic_change(self):
        text, units = self.units([words("alpha", 10)] * 3 + [words("beta", 10)] * 3)
        eng = self.engine(mn=10, mx=40, pct=100)
        spans = eng.chunk_spans(text, units)
        self.assertEqual(spans, [(units[0][0], units[2][1]), (units[3][0], units[5][1])])
        self.assertEqual((eng.stats["oversize_splits"], eng.stats["under_min"], eng.stats["over_max"]), (1, 0, 0))

    def test_split_prefers_cuts_that_keep_both_sides_above_min(self):
        # 가장 큰 거리는 처음 문장 뒤(5단어|45단어)지만 왼쪽이 min 미만이라 양쪽이 min 이상인 곳을 고른다
        text, units = self.units([words("alpha", 5)] + [words("beta", 15)] * 3)
        eng = self.engine(mn=10, mx=40, pct=100)
        spans = eng.chunk_spans(text, units)
        sizes = [count_words(text[s:e]) for s, e in spans]
        self.assertTrue(all(10 <= x <= 40 for x in sizes), sizes)

    def test_small_group_merges_forward_across_a_forced_boundary_and_is_counted(self):
        text, units = self.units([words("alpha", 3), words("beta", 12), words("beta", 12)])
        eng = self.engine(mn=10, mx=100, pct=100)
        spans = eng.chunk_spans(text, units, forced=[0])               # 첫 문장(3단어)을 강제로 떼어도 min 미만이라 합쳐진다
        self.assertEqual(spans, [(units[0][0], units[2][1])])
        self.assertEqual(eng.stats["merges_across_forced"], 1)

    def test_last_small_group_merges_backward(self):
        text, units = self.units([words("alpha", 12), words("beta", 12), words("gamma", 3)])
        eng = self.engine(mn=10, mx=100, pct=40)
        spans = eng.chunk_spans(text, units, forced=[1])
        self.assertEqual(count_words(text[spans[-1][0]:spans[-1][1]]) >= 10, True)

    def test_unmergeable_small_group_is_reported_not_hidden(self):
        # 앞 청크와 합치면 max를 넘는 짧은 마지막 그룹
        text, units = self.units([words("alpha", 28), words("beta", 3)])
        eng = self.engine(mn=10, mx=30, pct=100)
        spans = eng.chunk_spans(text, units, forced=[0])
        self.assertEqual(len(spans), 2)
        self.assertEqual(eng.stats["under_min"], 1)

    def test_single_unit_over_max_is_kept_and_not_counted_as_violation(self):
        text, units = self.units([words("alpha", 5), words("beta", 80), words("gamma", 5)])
        eng = self.engine(mn=3, mx=30, pct=100)
        spans = eng.chunk_spans(text, units, forced=[0, 1])
        big = (units[1][0], units[1][1])
        self.assertIn(big, spans)
        self.assertEqual(eng.stats["over_max"], 0)

    def test_chunks_are_ordered_non_overlapping_literal_slices_and_deterministic(self):
        text, units = self.units([words("alpha", 9)] * 5 + [words("beta", 9)] * 5)
        eng = self.engine(mn=10, mx=40, pct=80)
        spans = eng.chunk_spans(text, units)
        for a, b in zip(spans, spans[1:]):
            self.assertLessEqual(a[1], b[0])
        self.assertEqual("".join("".join(text[s:e].split()) for s, e in spans), "".join(text.split()))      # 내용 보존
        self.assertEqual(spans, eng.chunk_spans(text, units))

    def test_trivial_inputs(self):
        eng = self.engine(5, 50)
        self.assertEqual(eng.chunk_spans("", []), [])
        self.assertEqual(eng.chunk_spans("Only one.", [(0, 9)]), [(0, 9)])


class DPSizeControl(unittest.TestCase):
    """size_control="dp"(smart 전용): forced/max는 하드, min은 약한 벌점, 남은 작은 청크는 유사한 이웃과 합친다."""
    def engine(self, mn, mx, pct=100, min_weight=0.5, size_control="dp"):
        return SemanticEngine(make_stub(TOPIC_AXES)._client, count_words, percentile=pct, min_tokens=mn, max_tokens=mx,
                              size_control=size_control, min_weight=min_weight)

    def units(self, parts):
        text = "\n\n".join(parts)
        return text, sentence_spans(text)

    def prepared(self, eng, sizes):
        text, units = self.units([words("alpha", n) for n in sizes])
        eng._prepare_sizes(text, units)
        return text, units

    def brute_force(self, eng, n, d, tau, forced):
        best = None
        for mask in range(2 ** (n - 1)):
            cuts = {i for i in range(n - 1) if mask >> i & 1}
            if not forced <= cuts:
                continue
            bounds = [0] + [c + 1 for c in sorted(cuts)] + [n]
            segs = list(zip(bounds, bounds[1:]))
            if any(eng._size(a, b) > eng.max_t and b - a > 1 for a, b in segs):
                continue
            score = sum(0.0 if c in forced else d[c] - tau for c in cuts)
            score -= sum(eng.min_w * max(0, eng.min_t - eng._size(a, b)) / eng.min_t for a, b in segs)
            if best is None or score > best[0] + 1e-12:
                best = (score, segs)
        return best

    def test_worked_example_finds_the_only_partition_that_keeps_every_chunk_in_range(self):
        # 문서 설명용 예시: 그리디/재균형은 70토큰 청크가 남지만, DP는 앞쪽 경계까지 바꿔서 모두 100~200으로 만든다
        eng = self.engine(100, 200, min_weight=10)
        self.prepared(eng, [60, 50, 120, 40, 90, 80, 70])
        d = [0.10, 0.70, 0.20, 0.60, 0.15, 0.50]
        groups, score = eng._dp_groups(7, d, 0.30, set())
        self.assertEqual(groups, [(0, 2), (2, 3), (3, 5), (5, 7)])
        self.assertAlmostEqual(score, 0.15)
        self.assertEqual([eng._size(a, b) for a, b in groups], [110, 120, 130, 150])
        cuts = [i for i, dist in enumerate(d) if dist > 0.30]                  # 임계값 컷만 쓰면 마지막이 70토큰
        bounds = [0] + [c + 1 for c in cuts] + [7]
        self.assertEqual([eng._size(a, b) for a, b in zip(bounds, bounds[1:])], [110, 160, 170, 70])

    def test_dp_score_equals_brute_force_on_random_small_inputs(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            n = int(rng.integers(2, 9))
            sizes = [int(x) for x in rng.integers(5, 90, n)]
            d = [float(x) for x in rng.random(n - 1)]
            forced = {i for i in range(n - 1) if rng.random() < 0.2}
            eng = self.engine(int(rng.integers(20, 60)), int(rng.integers(90, 160)), min_weight=float(rng.choice([0.0, 0.5, 5.0])))
            self.prepared(eng, sizes)
            tau = float(np.percentile(d, 90))
            groups, score = eng._dp_groups(n, d, tau, forced)
            best = self.brute_force(eng, n, d, tau, forced)
            self.assertAlmostEqual(score, best[0], places=9, msg=(sizes, d, forced))
            self.assertTrue(all(not any(i in forced for i in range(a, b - 1)) for a, b in groups))     # forced는 안쪽에 없다
            self.assertEqual((groups[0][0], groups[-1][1]), (0, n))
            self.assertTrue(all(x[1] == y[0] for x, y in zip(groups, groups[1:])))

    def test_without_binding_constraints_it_cuts_exactly_where_distance_exceeds_threshold(self):
        eng = self.engine(1, 10_000)
        self.prepared(eng, [10] * 8)
        d = [0.1, 0.9, 0.2, 0.2, 0.8, 0.1, 0.3]
        groups, _ = eng._dp_groups(8, d, 0.5, set())
        self.assertEqual(groups, [(0, 2), (2, 5), (5, 8)])

    def test_forced_is_hard_and_single_unit_over_max_is_allowed(self):
        eng = self.engine(50, 100)
        self.prepared(eng, [5, 300, 5, 40, 40])
        groups, _ = eng._dp_groups(5, [0.0, 0.0, 0.0, 0.0], 0.5, {0, 1})
        self.assertEqual(groups, [(0, 1), (1, 2), (2, 5)])                      # forced 경계(0|1, 1|2)는 5토큰 청크가 되어도 자르고, 나머지(85)는 한 청크
        self.assertIn((1, 2), groups)                                            # 300토큰 단위는 그대로 한 청크
        self.assertEqual(groups[0], (0, 1))                                      # forced 경계는 작아도 자른다

    def test_small_chunk_joins_the_more_similar_neighbour_never_across_forced_or_above_max(self):
        eng = self.engine(10, 40)
        text, units = self.units([words("beta", 20), words("alpha", 5), words("alpha", 20)])
        eng._prepare_sizes(text, units)
        groups = [(0, 1), (1, 2), (2, 3)]
        self.assertEqual(eng._merge_similar(groups, set(), text, units), ([(0, 1), (1, 3)], 1))      # alpha 이웃과
        text, units = self.units([words("alpha", 20), words("alpha", 5), words("beta", 20)])
        eng._prepare_sizes(text, units)
        self.assertEqual(eng._merge_similar(groups, set(), text, units)[0], [(0, 2), (2, 3)])      # 앞과
        self.assertEqual(eng._merge_similar(groups, {0}, text, units)[0], [(0, 1), (1, 3)])        # 앞이 forced면 뒤와
        self.assertEqual(eng._merge_similar(groups, {0, 1}, text, units)[0], groups)               # 둘 다 forced면 그대로
        tight = self.engine(10, 22)
        tight._prepare_sizes(text, units)
        self.assertEqual(tight._merge_similar(groups, set(), text, units)[0], groups)              # 합치면 max 초과면 그대로

    def test_end_to_end_literal_slices_stats_and_greedy_default_unchanged(self):
        parts = [words("alpha", 60), words("alpha", 50), words("beta", 120), words("beta", 40), words("gamma", 90), words("gamma", 80), words("gamma", 70)]
        text, units = self.units(parts)
        dp = self.engine(100, 200, pct=60, min_weight=10)
        spans = dp.chunk_spans(text, units)
        self.assertEqual((spans[0][0], spans[-1][1]), (units[0][0], units[-1][1]))
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))
        self.assertEqual(dp.stats["chunks"], len(spans))
        for k in ("forced_cuts", "threshold_cuts", "size_cuts", "merges", "under_min", "over_max", "over_max_single_unit", "dp_score"):
            self.assertIn(k, dp.stats)
        self.assertEqual(spans, self.engine(100, 200, pct=60, min_weight=10).chunk_spans(text, units))   # 결정적
        greedy = SemanticEngine(make_stub(TOPIC_AXES)._client, count_words, percentile=60, min_tokens=100, max_tokens=200)
        self.assertEqual(greedy.size_control, "greedy")
        greedy.chunk_spans(text, units)
        self.assertIn("oversize_splits", greedy.stats)
        with self.assertRaises(ValueError):
            self.engine(10, 20, size_control="nope")


if __name__ == "__main__":
    unittest.main()
