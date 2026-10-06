"""구조 제약 규칙(units.py)과 smart = 엔진 + 규칙 구조의 테스트 - 가짜 임베딩과 공백 단위 토큰 카운터.
규칙: heading_forced(제목 앞 강제) / heading_glue(제목 뒤 금지) / table_atomic(표 한 덩어리) / caption_glue(표-캡션 금지)."""
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from chunking import splitters  # noqa: E402
from chunking.runner import tag_for  # noqa: E402
from chunking.semantic_core import sentence_spans  # noqa: E402
from chunking.units import ALL_RULES, NO_RULES, Rules, build_units, caption_glue, rules_for  # noqa: E402
from common.config import dataset_cfg  # noqa: E402
from common.docjson import render  # noqa: E402
from test_chunking import count_words, make_stub  # noqa: E402

TOPIC_AXES = ["alpha", "beta", "gamma"]


def words(topic, n):
    return " ".join([topic] * n) + "."


def make_doc(blocks):
    """[(type, text, extra)] -> 공통 JSON 문서 dict"""
    out = []
    for i, (typ, text, extra) in enumerate(blocks):
        b = {"id": f"b{i}", "type": typ, "text": text, "heading_path": [], "page": None}
        if typ == "heading":
            b["level"] = extra.get("level", 1)
        b.update({k: v for k, v in extra.items() if k != "level"})
        out.append(b)
    return {"doc_id": "d", "dataset": "t", "fmt": "pdf", "lang": "en", "blocks": out}


def units_for(doc, rules):
    text, pieces = render(doc)
    units, forced = build_units(text, pieces, caption_glue(doc, pieces), rules)
    return text, pieces, units, forced


def models():
    stub = make_stub(TOPIC_AXES)
    return {"embed_model": stub._client, "count": count_words}


class RulesObject(unittest.TestCase):
    def test_defaults_and_code(self):
        self.assertFalse(NO_RULES.any())
        self.assertTrue(ALL_RULES.any())
        self.assertEqual((NO_RULES.code(), ALL_RULES.code()), ("none", "HGTC"))
        self.assertEqual(Rules.from_dict({"table_atomic": True}).code(), "T")
        self.assertEqual(Rules.from_dict(ALL_RULES.as_dict()), ALL_RULES)

    def test_rules_for_uses_chunker_default_when_cfg_has_no_rules(self):
        self.assertEqual(rules_for("semantic", {}), NO_RULES)
        self.assertEqual(rules_for("smart", {}), ALL_RULES)
        self.assertEqual(rules_for("smart", {"rules": {"table_atomic": True}}), Rules(table_atomic=True))

    def test_config_semantic_is_all_off_and_smart_all_on(self):
        for ds in ("vectara", "allganize", "docx_corpus"):
            cfgs = dataset_cfg(ds)["chunkers"]
            self.assertEqual(Rules.from_dict(cfgs["semantic"]["rules"]), NO_RULES)
            self.assertEqual(Rules.from_dict(cfgs["smart"]["rules"]), ALL_RULES)
            for k in ("percentile", "min_tokens", "max_tokens", "buffer_size"):
                self.assertEqual(cfgs["semantic"][k], cfgs["smart"][k])        # 엔진 파라미터는 둘이 같다


class BuildUnits(unittest.TestCase):
    def doc_with_headings(self):
        return make_doc([("heading", "Intro", {"level": 1}), ("heading", "Scope", {"level": 2}),
                         ("paragraph", words("alpha", 5) + " " + words("alpha", 6), {}),
                         ("heading", "1. Method", {"level": 1}), ("paragraph", words("beta", 7), {})])

    def test_no_rules_equals_plain_sentence_split_of_the_whole_text(self):
        doc = self.doc_with_headings()
        text, pieces, units, forced = units_for(doc, NO_RULES)
        self.assertEqual(units, sentence_spans(text))
        self.assertEqual(forced, [])
        self.assertIn("# 1.", [text[s:e] for s, e in units])              # 제목도 평문처럼 마침표에서 나뉜다(LangChain과 같음)

    def test_heading_glue_attaches_headings_to_the_next_sentence(self):
        text, pieces, units, forced = units_for(self.doc_with_headings(), Rules(heading_glue=True))
        first = text[units[0][0]:units[0][1]]
        self.assertTrue(first.startswith("# Intro"))
        self.assertIn("## Scope", first)                                  # 연속 제목은 함께
        self.assertIn("alpha", first)                                     # 뒤따르는 첫 문장까지 한 단위
        self.assertEqual(forced, [])
        for s, e in units:                                                # 제목으로 끝나는 단위가 없다
            self.assertFalse(text[s:e].rstrip().endswith(("# Intro", "## Scope", "1. Method")))
        self.assertTrue(any(text[s:e].startswith("# 1. Method") for s, e in units))   # "1. Method"는 마침표에서 안 나뉜다

    def test_heading_off_keeps_heading_as_its_own_unit(self):
        text, pieces, units, forced = units_for(self.doc_with_headings(), Rules(heading_forced=True))
        self.assertEqual(text[units[0][0]:units[0][1]], "# Intro")        # 붙이지 않으면 제목이 혼자 단위

    def test_heading_forced_marks_boundaries_before_headings_only(self):
        text, pieces, units, forced = units_for(self.doc_with_headings(), ALL_RULES)
        starts = [text[units[i + 1][0]:units[i + 1][0] + 3] for i in forced]
        self.assertEqual(starts, ["# 1"])                                  # 문단 -> "# 1. Method" 앞 하나. 연속 제목 사이는 forced가 아님
        text, pieces, units, forced = units_for(self.doc_with_headings(), Rules(heading_glue=True))
        self.assertEqual(forced, [])                                       # 규칙이 꺼지면 forced도 없다

    def test_table_atomic_on_and_off(self):
        doc = make_doc([("paragraph", words("alpha", 5), {}), ("table", "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", {}),
                        ("paragraph", words("beta", 5), {})])
        text, pieces, units, _ = units_for(doc, Rules(table_atomic=True))
        table = next(p for p in pieces if p.kind == "table")
        self.assertIn((table.start, table.end), units)
        self.assertEqual(len(units), 3)
        text, pieces, units, _ = units_for(doc, NO_RULES)
        self.assertEqual(len(units), 2 + 4)                                # 표가 행 4개 단위로 쪼개진다

    def test_caption_glue_merges_caption_and_table_in_both_orders(self):
        doc = make_doc([("paragraph", words("alpha", 5), {}),
                        ("paragraph", "Table 1 shows results.", {"role": "caption", "for": "b2"}),
                        ("table", "| a | b |\n|---|---|\n| 1 | 2 |", {}),
                        ("paragraph", "Table 2 shows more.", {"role": "caption", "for": "b2"}),
                        ("paragraph", words("beta", 5), {})])
        text, pieces, units, _ = units_for(doc, Rules(table_atomic=True, caption_glue=True))
        merged = [text[s:e] for s, e in units if "Table 1" in text[s:e]]
        self.assertEqual(len(merged), 1)
        self.assertIn("| a | b |", merged[0])                               # 캡션(앞)과 표가 한 단위
        self.assertTrue(any("| a | b |" in text[s:e] and "Table 2" in text[s:e] for s, e in units))   # 캡션(뒤)도 합쳐짐: 세 조각이 한 덩어리
        text, pieces, units_off, _ = units_for(doc, Rules(table_atomic=True))
        self.assertGreater(len(units_off), len(units))


class SmartEqualsEnginePlusRules(unittest.TestCase):
    def cfg(self, rules=None):
        c = {"percentile": 90, "min_tokens": 10, "max_tokens": 80}
        if rules is not None:
            c["rules"] = rules
        return c

    def run_smart(self, doc, chunker="smart", rules=None):
        return splitters.chunk(doc, chunker, self.cfg(rules), models())

    def paper(self):
        return make_doc([("heading", "One", {}), ("paragraph", words("alpha", 12) + " " + words("alpha", 13), {}),
                         ("paragraph", words("alpha", 14), {}),
                         ("heading", "Two", {}), ("paragraph", words("beta", 15) + " " + words("beta", 16), {}),
                         ("table", "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |", {}), ("paragraph", words("beta", 17), {}),
                         ("heading", "Three", {}), ("paragraph", words("gamma", 18) + " " + words("gamma", 19), {})])

    def test_smart_with_all_rules_off_equals_semantic(self):
        doc = self.paper()
        smart_off, _ = self.run_smart(doc, "smart", rules=NO_RULES.as_dict())
        semantic, _ = self.run_smart(doc, "semantic")
        self.assertEqual([(c["start"], c["end"]) for c in smart_off], [(c["start"], c["end"]) for c in semantic])
        smart_on, _ = self.run_smart(doc, "smart")                         # 헛통과 방지: 이 문서에서는 규칙이 실제로 결과를 바꾼다
        self.assertNotEqual([(c["start"], c["end"]) for c in smart_on], [(c["start"], c["end"]) for c in semantic])

    def test_semantic_with_all_rules_on_equals_smart(self):
        doc = self.paper()
        sem_on, _ = self.run_smart(doc, "semantic", rules=ALL_RULES.as_dict())
        smart, _ = self.run_smart(doc, "smart")
        self.assertEqual([(c["start"], c["end"]) for c in sem_on], [(c["start"], c["end"]) for c in smart])

    def test_on_a_plain_text_document_rules_change_nothing(self):
        doc = make_doc([("paragraph", words("alpha", 9 + i), {}) for i in range(6)] + [("paragraph", words("beta", 9 + i), {}) for i in range(6)])
        a, _ = self.run_smart(doc, "smart")
        b, _ = self.run_smart(doc, "semantic")
        self.assertEqual([(c["start"], c["end"]) for c in a], [(c["start"], c["end"]) for c in b])

    def test_smart_structure_guarantees(self):
        doc = self.paper()
        chunks, info = self.run_smart(doc)
        text, pieces = render(doc)
        types = {b["id"]: b["type"] for b in doc["blocks"]}
        self.assertTrue(all(types[c["block_ids"][-1]] != "heading" for c in chunks))     # 제목으로 끝나는 청크 없음
        table = next(p for p in pieces if p.kind == "table")
        holders = [c for c in chunks if c["start"] <= table.start < c["end"]]
        self.assertEqual(len(holders), 1)
        self.assertGreaterEqual(holders[0]["end"], table.end)                              # 표는 한 청크 안에 통째로
        self.assertGreaterEqual(info["smart_forced_cuts"], 1)
        kept = re.sub(r"\s+", "", "".join(c["text"] for c in chunks))
        self.assertEqual(kept, re.sub(r"\s+", "", text))                                   # 내용 보존

    def test_semantic_splits_the_table_but_smart_does_not(self):
        doc = make_doc([("paragraph", words("alpha", 30), {}), ("table", "| k | v |\n" + "\n".join(f"| r{i} | {i} |" for i in range(40)), {}),
                        ("paragraph", words("alpha", 30), {})])
        text, pieces = render(doc)
        table = next(p for p in pieces if p.kind == "table")
        cfg = {"percentile": 90, "min_tokens": 5, "max_tokens": 30}
        sem, _ = splitters.chunk(doc, "semantic", cfg, models())
        smart, _ = splitters.chunk(doc, "smart", cfg, models())
        inside = lambda cs: [c for c in cs if c["start"] < table.end and c["end"] > table.start]
        self.assertGreater(len(inside(sem)), 1)                                           # semantic: 표가 여러 청크에 걸침
        self.assertEqual(len(inside(smart)), 1)                                           # smart: 표 하나가 한 청크에만 걸침
        self.assertEqual((inside(smart)[0]["start"], inside(smart)[0]["end"]), (table.start, table.end))   # max를 넘어도 단독 청크

    def test_short_section_merges_across_heading_when_below_min(self):
        doc = make_doc([("heading", "A", {}), ("paragraph", words("alpha", 30), {}), ("heading", "B", {}), ("paragraph", words("beta", 3), {})])
        chunks, info = splitters.chunk(doc, "smart", {"percentile": 90, "min_tokens": 10, "max_tokens": 80}, models())
        self.assertEqual(len(chunks), 1)                                                   # B 섹션(5단어)은 min 미만이라 앞과 합쳐짐
        self.assertEqual(info["smart_merges_across_forced"], 1)

    def test_tag_distinguishes_ablated_rules(self):
        self.assertEqual(tag_for("smart", self.cfg()), "smart__p90-min10-max80")
        self.assertEqual(tag_for("smart", self.cfg({"table_atomic": True})), "smart__p90-min10-max80-rT")
        self.assertEqual(tag_for("semantic", self.cfg()), "semantic__p90-min10-max80")


if __name__ == "__main__":
    unittest.main()
