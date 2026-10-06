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


class Rebalance(unittest.TestCase):
    """rebalance=True(smart 전용): min 미만 그룹을 forced가 아닌 이웃 중 유사한 쪽과 합치고, max를 넘으면 합친 안에서 다시 자른다."""
    def engine(self, mn, mx, pct=100, rebalance=True):
        return SemanticEngine(make_stub(TOPIC_AXES)._client, count_words, percentile=pct, min_tokens=mn, max_tokens=mx, rebalance=rebalance)

    def units(self, parts):
        text = "\n\n".join(parts)
        return text, sentence_spans(text)

    def groups_after(self, eng, text, units, groups, forced=()):
        eng._prepare_sizes(text, units)
        d = consecutive_distances(eng.embed, [text[s:e] for s, e in units])
        return eng._rebalance_small(groups, set(forced), d, text, units)

    def test_small_group_that_cannot_merge_is_rebalanced_into_its_neighbour(self):
        # 8단어 그룹 + 35단어 그룹 = 43 > max 40 -> 기본 합치기는 못 하고(min 미달 보고), 재균형은 경계를 옮겨 둘 다 min 이상으로 만든다
        parts = [words("alpha", 8), words("beta", 15), words("beta", 10), words("beta", 10)]
        text, units = self.units(parts)
        # 의미 컷(순수 거리)만으로 첫 문장을 떼는 임계값
        eng_off = self.engine(10, 40, pct=60, rebalance=False)
        eng_on = self.engine(10, 40, pct=60, rebalance=True)
        off, on = eng_off.chunk_spans(text, units), eng_on.chunk_spans(text, units)
        self.assertEqual(eng_off.stats["under_min"], 1)
        self.assertEqual((eng_on.stats["under_min"], eng_on.stats["rebalanced"]), (0, 1))
        sizes = [count_words(text[s:e]) for s, e in on]
        self.assertTrue(all(10 <= x <= 40 for x in sizes), sizes)
        self.assertEqual((on[0][0], on[-1][1]), (units[0][0], units[-1][1]))
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(on, on[1:])))           # 겹침 없음

    def test_neighbour_is_chosen_by_chunk_similarity(self):
        text, units = self.units([words("beta", 20), words("alpha", 5), words("alpha", 20)])
        got, _ = self.groups_after(self.engine(10, 40), text, units, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(got, [(0, 1), (1, 3)])                                  # alpha 조각은 alpha 이웃(뒤)과
        text, units = self.units([words("alpha", 20), words("alpha", 5), words("beta", 20)])
        got, _ = self.groups_after(self.engine(10, 40), text, units, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(got, [(0, 2), (2, 3)])                                  # 이번에는 앞과

    def test_forced_boundaries_are_never_crossed(self):
        text, units = self.units([words("alpha", 20), words("alpha", 5), words("alpha", 20)])
        groups = [(0, 1), (1, 2), (2, 3)]
        got, _ = self.groups_after(self.engine(10, 40), text, units, groups, forced=[1])      # 작은 조각 뒤가 forced -> 앞과
        self.assertEqual(got, [(0, 2), (2, 3)])
        got, _ = self.groups_after(self.engine(10, 40), text, units, groups, forced=[0])      # 작은 조각 앞이 forced -> 뒤와
        self.assertEqual(got, [(0, 1), (1, 3)])
        got, _ = self.groups_after(self.engine(10, 40), text, units, groups, forced=[0, 1])   # 양쪽 다 forced -> 그대로 두고 보고
        self.assertEqual(got, groups)
        eng = self.engine(10, 40)
        eng.chunk_spans(text, units, forced=[0, 1])
        self.assertEqual((eng.stats["merges_across_forced"], eng.stats["under_min"]), (0, 1))

    def test_default_engine_is_unchanged_and_single_unit_over_max_is_reported(self):
        text, units = self.units([words("alpha", 5), words("beta", 80), words("gamma", 5)])
        eng = self.engine(3, 30, rebalance=False)
        eng.chunk_spans(text, units, forced=[0, 1])
        self.assertEqual((eng.stats["rebalanced"], eng.stats["over_max"], eng.stats["over_max_single_unit"]), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
