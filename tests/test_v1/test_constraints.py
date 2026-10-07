"""구조 제약 규칙(units.py)과 smart = 엔진 + 규칙 구조의 테스트 - 가짜 임베딩과 공백 단위 토큰 카운터.
규칙: heading_forced(제목 앞 강제) / heading_glue(제목 뒤 금지) / table_atomic(표 한 덩어리) / caption_glue(표-캡션 금지)."""
import re
import unittest


from support import TOPIC_AXES, make_stub, count_words, words
from chunking.v1 import splitters
from chunking.v1.runner import tag_for
from chunking.sizectl.semantic_core import sentence_spans
from chunking.v1.units import SMART_RULES, NO_RULES, Rules, build_units, caption_glue, rules_for
from common.config import dataset_cfg
from common.docjson import render



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
        self.assertTrue(SMART_RULES.any())
        self.assertEqual((NO_RULES.code(), SMART_RULES.code()), ("none", "GTCS"))          # smart 기본: 제목 붙임·표·캡션·섹션 분할(S). 제목 앞 강제(H)와 문단 단위(P)는 어블레이션용
        self.assertFalse(SMART_RULES.heading_forced or SMART_RULES.para_atomic)
        self.assertTrue(SMART_RULES.section_split)
        self.assertEqual(Rules.from_dict({"table_atomic": True}).code(), "T")
        self.assertEqual(Rules.from_dict(SMART_RULES.as_dict()), SMART_RULES)

    def test_rules_for_uses_chunker_default_when_cfg_has_no_rules(self):
        self.assertEqual(rules_for("semantic", {}), NO_RULES)
        self.assertEqual(rules_for("smart", {}), SMART_RULES)
        self.assertEqual(rules_for("smart", {"rules": {"table_atomic": True}}), Rules(table_atomic=True))

    def test_config_semantic_is_all_off_and_smart_all_on(self):
        for ds in ("vectara", "allganize", "docx_corpus"):
            cfgs = dataset_cfg(ds)["chunkers"]
            self.assertEqual(Rules.from_dict(cfgs["semantic"]["rules"]), NO_RULES)
            self.assertEqual(Rules.from_dict(cfgs["smart"]["rules"]), SMART_RULES)
            for k in ("percentile", "min_tokens", "max_tokens", "buffer_size"):
                self.assertEqual(cfgs["semantic"][k], cfgs["smart"][k])        # 엔진 파라미터는 둘이 같다
            self.assertEqual((cfgs["semantic"]["size_control"], cfgs["smart"]["size_control"]), ("dp", "dp"))   # 크기 제어는 둘 다 DP(차이는 규칙뿐)
            self.assertNotIn("min_weight", cfgs["smart"])                                                  # min은 사전식 하드라 튜닝 값이 없다


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
        with_forced = Rules(**{**SMART_RULES.as_dict(), "heading_forced": True})            # 어블레이션: 제목 앞 강제 절단을 켠 smart
        text, pieces, units, forced = units_for(self.doc_with_headings(), with_forced)
        starts = [text[units[i + 1][0]:units[i + 1][0] + 3] for i in forced]
        self.assertEqual(starts, ["# 1"])                                  # 문단 -> "# 1. Method" 앞 하나. 연속 제목 사이는 forced가 아님
        text, pieces, units, forced = units_for(self.doc_with_headings(), SMART_RULES)
        self.assertEqual(forced, [])                                       # smart 기본에는 forced가 없다
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


class ParaAtomic(unittest.TestCase):
    """para_atomic(선택 규칙, 기본 꺼짐): 문단 하나를 단위 하나로 -> 문단 단위 임베딩. max를 넘는 문단은 문장으로 푼다."""
    def doc(self):
        para = lambda t, n1, n2, n3: " ".join([words(t, n1), words(t, n2), words(t, n3)])      # 문장 3개짜리 문단
        return make_doc([("heading", "Intro", {"level": 1}), ("paragraph", para("alpha", 5, 6, 7), {}),
                         ("paragraph", para("beta", 8, 9, 10), {}), ("paragraph", words("beta", 4), {})])

    def build(self, rules, count=count_words, max_tokens=None):
        doc = self.doc()
        text, pieces = render(doc)
        units, forced = build_units(text, pieces, caption_glue(doc, pieces), rules, count=count, max_tokens=max_tokens)
        return text, pieces, units, forced

    def test_off_by_default_for_both_and_code_letter(self):
        self.assertFalse(SMART_RULES.para_atomic)                                           # smart도 semantic처럼 문장 단위가 기본
        self.assertFalse(NO_RULES.para_atomic)
        self.assertEqual(SMART_RULES.code(), "GTCS")                                        # 기본 smart의 파일 이름 코드(기본 세트라 이름에는 안 붙음)
        self.assertEqual(Rules.from_dict({**SMART_RULES.as_dict(), "para_atomic": True}).code(), "GTCPS")  # 문단 단위 smart(어블레이션)
        self.assertEqual(rules_for("smart", {}), SMART_RULES)

    def test_paragraph_becomes_one_unit_and_off_keeps_sentences(self):
        rules_on = Rules(para_atomic=True, heading_glue=True)
        text, pieces, units_on, _ = self.build(rules_on)
        text, pieces, units_off, _ = self.build(Rules(heading_glue=True))
        paras = [p for p in pieces if p.kind == "para"]
        self.assertEqual(len(units_off), 1 + 3 + 3 + 1 - 1)                               # 제목+첫 문장이 붙고 나머지는 문장마다(7)
        self.assertEqual(len(units_on), 3)                                                # 문단마다 단위 하나(제목은 첫 문단에 붙음)
        self.assertEqual([text[s:e] for s, e in units_on][1:], [text[p.start:p.end] for p in paras[1:]])
        self.assertTrue(text[units_on[0][0]:units_on[0][1]].startswith("# Intro"))      # heading_glue와 함께 동작

    def test_paragraph_over_max_is_split_into_sentences(self):
        text, pieces, units, _ = self.build(Rules(para_atomic=True), max_tokens=20)       # 둘째 문단은 27단어 > 20, 첫째 18단어
        paras = [p for p in pieces if p.kind == "para"]
        inside = lambda p: [u for u in units if u[0] >= p.start and u[1] <= p.end]
        self.assertEqual(len(inside(paras[0])), 1)                                        # 18 <= 20: 통째로
        self.assertEqual(len(inside(paras[1])), 3)                                        # 27 > 20: 문장 3개
        self.assertEqual(len(inside(paras[2])), 1)

    def test_chunks_never_cut_inside_a_paragraph_when_every_paragraph_fits(self):
        doc = make_doc([("paragraph", words("alpha", 8) + " " + words("alpha", 9), {}) for _ in range(6)]
                       + [("paragraph", words("beta", 8) + " " + words("beta", 9), {}) for _ in range(6)])
        cfg = {"percentile": 90, "min_tokens": 20, "max_tokens": 60, "rules": {"para_atomic": True}}
        chunks, _ = splitters.chunk(doc, "smart", cfg, models())
        text, pieces = render(doc)
        starts = {p.start for p in pieces}
        self.assertTrue(all(c["start"] in starts for c in chunks))                        # 청크는 문단 시작에서만 시작
        sent, _ = splitters.chunk(doc, "smart", {**cfg, "rules": {}}, models())           # 규칙이 없으면 문장 사이에서도 자를 수 있다(비교)
        self.assertNotEqual(tag_for("smart", cfg), tag_for("smart", {**cfg, "rules": {}}))
        self.assertTrue(tag_for("smart", cfg).endswith("-rP"))


class SmartEqualsEnginePlusRules(unittest.TestCase):
    def cfg(self, rules=None, size_control=None):
        c = {"percentile": 90, "min_tokens": 10, "max_tokens": 80}
        if rules is not None:
            c["rules"] = rules
        if size_control is not None:
            c["size_control"] = size_control
        return c

    def run_smart(self, doc, chunker="smart", rules=None, size_control=None):
        return splitters.chunk(doc, chunker, self.cfg(rules, size_control), models())

    def paper(self):
        return make_doc([("heading", "One", {}), ("paragraph", words("alpha", 12) + " " + words("alpha", 13), {}),
                         ("paragraph", words("alpha", 14), {}),
                         ("heading", "Two", {}), ("paragraph", words("beta", 15) + " " + words("beta", 16), {}),
                         ("table", "| k | v |\n|---|---|\n| a | 1 |\n| b | 2 |", {}), ("paragraph", words("beta", 17), {}),
                         ("heading", "Three", {}), ("paragraph", words("gamma", 18) + " " + words("gamma", 19), {})])

    def test_smart_with_all_rules_off_equals_semantic(self):
        doc = self.paper()
        smart_off, _ = self.run_smart(doc, "smart", rules=NO_RULES.as_dict())                            # 크기 제어(둘 다 dp)가 같을 때 같다
        semantic, _ = self.run_smart(doc, "semantic")
        self.assertEqual([(c["start"], c["end"]) for c in smart_off], [(c["start"], c["end"]) for c in semantic])
        smart_on, _ = self.run_smart(doc, "smart")                         # 헛통과 방지: 이 문서에서는 규칙이 실제로 결과를 바꾼다
        self.assertNotEqual([(c["start"], c["end"]) for c in smart_on], [(c["start"], c["end"]) for c in semantic])

    def test_semantic_with_all_rules_on_equals_smart(self):
        doc = self.paper()
        sem_on, _ = self.run_smart(doc, "semantic", rules=SMART_RULES.as_dict())                           # 규칙과 크기 제어가 같으면 같다
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
        self.assertEqual(info["smart_forced_cuts"], 0)                                     # smart 기본에는 제목 앞 강제 절단이 없다
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

    def test_short_section_is_merged_by_default_and_the_merge_is_reported(self):
        doc = make_doc([("heading", "A", {}), ("paragraph", words("alpha", 30), {}), ("heading", "B", {}), ("paragraph", words("beta", 3), {})])
        cfg = {"percentile": 90, "min_tokens": 10, "max_tokens": 80}
        chunks, info = splitters.chunk(doc, "smart", cfg, models())                          # 기본(섹션 분할): 짧은 섹션 B가 min 때문에 A와 합쳐진다(C안)
        self.assertEqual((len(chunks), info["smart_under_min"]), (1, 0))
        self.assertEqual((info["smart_orig_sections"], info["smart_section_merges"], info["smart_cross_section_chunks"]), (2, 1, 1))   # 섞인 청크는 보고된다

    def test_heading_forced_ablation_yields_to_min_and_greedy_also_crosses_it(self):
        doc = make_doc([("heading", "A", {}), ("paragraph", words("alpha", 30), {}), ("heading", "B", {}), ("paragraph", words("beta", 3), {})])
        cfg = {"percentile": 90, "min_tokens": 10, "max_tokens": 80}
        no_sections = {**SMART_RULES.as_dict(), "section_split": False, "heading_forced": True}   # 어블레이션: 섹션 분할 없이 제목 앞 강제만
        chunks, info = splitters.chunk(doc, "smart", {**cfg, "rules": no_sections}, models())        # DP: forced는 보너스일 뿐이라 min이 우선 - 짧은 절이 합쳐진다
        self.assertEqual(len(chunks), 1)
        self.assertEqual((info["smart_under_min"], info["smart_forced_cuts"]), (0, 0))
        chunks, info = splitters.chunk(doc, "smart", {**cfg, "rules": no_sections, "size_control": "greedy"}, models())   # 그리디: min 때문에 제목 경계를 넘어 합침
        self.assertEqual(len(chunks), 1)
        self.assertEqual(info["smart_merges_across_forced"], 1)
        with self.assertRaises(ValueError):                                                         # 섹션 분할은 DP 전용
            splitters.chunk(doc, "smart", {**cfg, "size_control": "greedy"}, models())

    def test_tag_distinguishes_ablated_rules(self):
        self.assertEqual(tag_for("smart", self.cfg()), "smart__p90-min10-max80")
        self.assertEqual(tag_for("smart", self.cfg({"table_atomic": True})), "smart__p90-min10-max80-rT")
        self.assertEqual(tag_for("semantic", self.cfg()), "semantic__p90-min10-max80")
        self.assertEqual(tag_for("smart", self.cfg(size_control="greedy")), "smart__p90-min10-max80-scgreedy")      # 어블레이션은 이름에 남는다
        self.assertEqual(tag_for("semantic", self.cfg(size_control="greedy")), "semantic__p90-min10-max80-scgreedy")


if __name__ == "__main__":
    unittest.main()
