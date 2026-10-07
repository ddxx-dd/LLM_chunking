"""최상위 섹션 분할(chunking/sections.py)과 섹션 묶음 처리(SemanticEngine.chunk_groups) 테스트 - 가짜 임베딩, 공백 단위 토큰 카운터."""
import unittest

from support import TOPIC_AXES, make_stub, count_words, words
from chunking.v1 import splitters
from chunking.v1.sections import merge_short_sections, section_level, split_sections
from chunking.sizectl.semantic_core import SemanticEngine, sentence_spans
from common.docjson import render


def make_doc(blocks):
    out = []
    for i, (typ, text, extra) in enumerate(blocks):
        b = {"id": f"b{i}", "type": typ, "text": text, "heading_path": [], "page": None}
        if typ == "heading":
            b["level"] = extra.get("level", 1)
        out.append(b)
    return {"doc_id": "d", "dataset": "t", "fmt": "pdf", "lang": "en", "blocks": out}


H = lambda text, level=1: ("heading", text, {"level": level})
P = lambda topic, n: ("paragraph", words(topic, n), {})
EMB = make_stub(TOPIC_AXES)._client


class SectionLevel(unittest.TestCase):
    def level(self, blocks):
        doc = make_doc(blocks)
        return section_level(doc, render(doc)[1])

    def test_shallowest_level_with_at_least_two_headings(self):
        self.assertEqual(self.level([H("A"), P("alpha", 5), H("B"), P("beta", 5)]), 1)
        self.assertEqual(self.level([H("Title", 1), H("A", 2), P("alpha", 5), H("B", 2), P("beta", 5)]), 2)      # 문서 제목만 #이면 ##로 내려감
        self.assertEqual(self.level([H("Title", 1), H("A", 2), H("A1", 3), H("A2", 3), P("alpha", 5)]), 3)
        self.assertIsNone(self.level([H("Title", 1), P("alpha", 5), H("A", 2), P("beta", 5)]))              # 수준마다 하나뿐
        self.assertIsNone(self.level([P("alpha", 5), P("beta", 5)]))                                      # 제목 없음

    def test_split_ranges_cover_every_piece_and_start_at_headings(self):
        doc = make_doc([H("Title", 1), P("alpha", 8), H("A", 2), P("alpha", 9), H("A1", 3), P("alpha", 7), H("B", 2), P("beta", 6)])
        text, pieces = render(doc)
        sections, lvl = split_sections(doc, pieces)
        self.assertEqual(lvl, 2)
        self.assertEqual(sections, [(0, 2), (2, 6), (6, 8)])                                              # 제목+머리말 / A(+A1 포함) / B
        self.assertEqual([p for lo, hi in sections for p in range(lo, hi)], list(range(len(pieces))))

    def test_preamble_before_the_first_heading_is_its_own_section(self):
        doc = make_doc([P("alpha", 8), H("A"), P("alpha", 9), H("B"), P("beta", 6)])
        text, pieces = render(doc)
        self.assertEqual(split_sections(doc, pieces)[0], [(0, 1), (1, 3), (3, 5)])

    def test_no_split_when_there_is_no_usable_level(self):
        doc = make_doc([H("Only", 1), P("alpha", 8), P("alpha", 9)])
        text, pieces = render(doc)
        self.assertEqual(split_sections(doc, pieces), ([(0, 3)], None))


class MergeShortSections(unittest.TestCase):
    def merge(self, blocks, mn, mx):
        doc = make_doc(blocks)
        text, pieces = render(doc)
        sections, _ = split_sections(doc, pieces)
        groups, merges = merge_short_sections(text, pieces, sections, EMB, count_words, mn, mx)
        return sections, groups, merges

    def test_short_section_joins_the_more_similar_neighbour(self):
        blocks = [H("A"), P("beta", 40), H("B"), P("alpha", 3), H("C"), P("alpha", 40)]
        sections, groups, merges = self.merge(blocks, mn=20, mx=100)
        self.assertEqual((sections, groups, merges), ([(0, 2), (2, 4), (4, 6)], [(0, 2), (2, 6)], 1))        # B(alpha)는 C(alpha)와
        blocks = [H("A"), P("alpha", 40), H("B"), P("alpha", 3), H("C"), P("beta", 40)]
        self.assertEqual(self.merge(blocks, 20, 100)[1], [(0, 4), (4, 6)])                                  # 이번에는 A와

    def test_merge_never_exceeds_max_and_unmergeable_stays(self):
        blocks = [H("A"), P("alpha", 60), H("B"), P("alpha", 3), H("C"), P("alpha", 60)]
        sections, groups, merges = self.merge(blocks, mn=20, mx=64)
        self.assertEqual((groups, merges), (sections, 0))                                                    # 어느 쪽과 합쳐도 max 초과 -> 그대로

    def test_sections_that_already_fit_are_untouched(self):
        blocks = [H("A"), P("alpha", 30), H("B"), P("beta", 30)]
        sections, groups, merges = self.merge(blocks, mn=20, mx=100)
        self.assertEqual((groups, merges), (sections, 0))


class ChunkGroups(unittest.TestCase):
    def engine(self, mn=10, mx=60, pct=90, sc="dp"):
        return SemanticEngine(EMB, count_words, percentile=pct, min_tokens=mn, max_tokens=mx, size_control=sc)

    def groups(self, parts_per_group):
        """문장 목록들 -> (평문, [(단위 목록, forced 집합)])  묶음마다 자기 문장 구간만 단위로 가진다."""
        text, groups = "", []
        for parts in parts_per_group:
            start = len(text) + (2 if text else 0)
            text += ("\n\n" if text else "") + "\n\n".join(parts)
            groups.append(([u for u in sentence_spans(text) if u[0] >= start], set()))
        return text, groups

    def test_no_chunk_crosses_a_group_boundary_and_stats_count_single_unit_groups(self):
        text, groups = self.groups([[words("alpha", 12), words("alpha", 13), words("beta", 14)], [words("gamma", 4)], [words("beta", 15), words("beta", 16)]])
        eng = self.engine(mn=10, mx=60)
        spans = eng.chunk_groups(text, groups)
        bounds = [g[0][0][0] for g in groups[1:]]
        self.assertTrue(all(not (s < b < e) for s, e in spans for b in bounds))
        self.assertEqual(eng.stats["section_groups"], 3)
        self.assertEqual(eng.stats["under_min"], 1)                                                           # 단위 하나짜리 묶음(4단어)도 min 미달로 센다
        self.assertEqual((spans[0][0], spans[-1][1]), (groups[0][0][0][0], groups[-1][0][-1][1]))
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))

    def test_threshold_is_taken_over_all_groups_not_per_group(self):
        a = [words("alpha", 10)] * 3 + [words("beta", 10)] * 3                                     # 의미 경계가 뚜렷한 묶음(큰 거리 하나)
        b = [" ".join(["gamma"] * 8 + ["beta"] * k) + "." for k in range(1, 7)]                    # 작은 변화만 있는 묶음
        text, groups = self.groups([a, b])
        eng = self.engine(mn=5, mx=200, pct=90)
        spans = eng.chunk_groups(text, groups)
        in_b = [sp for sp in spans if sp[0] >= groups[1][0][0][0]]
        self.assertEqual(len(in_b), 1)                                                              # 문서 기준 τ: 작은 변화는 경계가 아니다
        alone = self.engine(mn=5, mx=200, pct=90).chunk_groups(text, [groups[1]])
        self.assertGreater(len(alone), 1)                                                           # 대조군: 그 묶음만으로 τ를 구하면 노이즈에서 잘린다

    def test_requires_dp(self):
        text, groups = self.groups([[words("alpha", 12), words("beta", 13)]])
        with self.assertRaises(ValueError):
            self.engine(sc="greedy").chunk_groups(text, groups)


class SmartSectionIntegration(unittest.TestCase):
    cfg = {"percentile": 90, "min_tokens": 10, "max_tokens": 60}

    def models(self):
        return {"embed_model": EMB, "count": count_words}

    def test_chunks_stay_inside_sections_when_every_section_fits(self):
        doc = make_doc([H("A"), P("alpha", 20), P("alpha", 22), H("B"), P("beta", 20), P("beta", 25), H("C"), P("gamma", 30)])
        text, pieces = render(doc)
        chunks, info = splitters.chunk(doc, "smart", self.cfg, self.models())
        starts = [p.start for p in pieces if p.kind == "heading"][1:]
        self.assertTrue(all(not (c["start"] < b < c["end"]) for c in chunks for b in starts))
        self.assertEqual((info["smart_orig_sections"], info["smart_section_merges"], info["smart_cross_section_chunks"]), (3, 0, 0))
        self.assertTrue(all(c["text"].startswith(("# ", "## ")) or c["start"] not in starts for c in chunks))

    def test_semantic_is_not_sectioned(self):
        doc = make_doc([H("A"), P("alpha", 20), H("B"), P("beta", 20)])
        chunks, info = splitters.chunk(doc, "semantic", self.cfg, self.models())
        self.assertNotIn("semantic_orig_sections", info)


if __name__ == "__main__":
    unittest.main()
