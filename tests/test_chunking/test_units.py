"""chunking.units: 단위 만들기와 구조 규칙 3종. 옛 v1 build_units(같은 규칙 3개만 켠 것)와 같은 결과여야 한다."""
import unittest

import support
from support import FIX
from chunking import base, units
from chunking.v1 import units as old
from common.docjson import load_doc, render


def synthetic_doc():
    return {"doc_id": "syn", "dataset": "docx_corpus", "fmt": "docx", "lang": "en", "blocks": [
        {"id": "h1", "type": "heading", "level": 1, "text": "1 Intro"},
        {"id": "h2", "type": "heading", "level": 2, "text": "1.1 Sub"},
        {"id": "p1", "type": "paragraph", "text": "First sentence here. Second sentence here."},
        {"id": "t1", "type": "table", "text": "| a | b |\n| 1 | 2 |"},
        {"id": "c1", "type": "paragraph", "role": "caption", "for": "t1", "text": "Table 1: caption text."},
        {"id": "p2", "type": "paragraph", "text": "After the table. Last one."}]}


class TestUnits(unittest.TestCase):
    def ctx(self, doc):
        return base.DocContext(doc)

    def test_sent_is_plain_sentences(self):
        c = self.ctx(load_doc(FIX / "example_a.json"))
        self.assertEqual(units.make_units(c, "sent"), base.sentence_spans(c.text))

    def test_sentg_same_as_v1_with_three_rules(self):
        rules = old.Rules(heading_glue=True, table_atomic=True, caption_glue=True)
        docs = [load_doc(FIX / f"example_{k}.json") for k in "ac"] + [synthetic_doc()]
        for doc in docs:
            c = self.ctx(doc)
            want, _ = old.build_units(c.text, c.pieces, old.caption_glue(doc, c.pieces), rules)
            self.assertEqual(units.make_units(c, "sentg"), want, doc["doc_id"])

    def test_caption_glue_same_as_v1(self):
        for doc in (load_doc(FIX / "example_a.json"), synthetic_doc()):
            _, pieces = render(doc)
            self.assertEqual(units.caption_glue(doc, pieces), old.caption_glue(doc, pieces))

    def test_rules_on_synthetic(self):
        doc = synthetic_doc()
        c = self.ctx(doc)
        u = units.make_units(c, "sentg")
        texts = [c.text[s:e] for s, e in u]
        # 연속 제목 두 개와 첫 문장이 한 단위(제목 바로 뒤는 못 자름)
        self.assertEqual(texts[0], "# 1 Intro\n\n## 1.1 Sub\n\nFirst sentence here.")
        self.assertEqual(texts[1], "Second sentence here.")
        # 표와 캡션이 한 단위(표는 통째로, 캡션과 붙음)
        table_units = [t for t in texts if "| a | b |" in t]
        self.assertEqual(len(table_units), 1)
        self.assertIn("Table 1: caption text.", table_units[0])
        # 표 앞뒤 문장은 따로
        self.assertIn("After the table.", texts)

    def test_units_are_disjoint_ordered(self):
        for k in "ac":
            c = self.ctx(load_doc(FIX / f"example_{k}.json"))
            for kind in ("sent", "sentg"):
                prev = 0
                for s, e in units.make_units(c, kind):
                    self.assertTrue(prev <= s < e <= len(c.text))
                    prev = e

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            units.make_units(self.ctx(synthetic_doc()), "para")


if __name__ == "__main__":
    unittest.main()
