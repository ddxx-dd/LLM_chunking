"""docx_corpus 질문 자동 필터(qa_build/docx_filters) 단위 테스트 - 작은 합성 구간."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from collections import Counter

from support import HERE
from qa_build import docx_filters as F

SPAN = ("The Zephyr Board must publish the annual audit within 45 days of the financial year end. "
        "Late filings attract a penalty of $1 200 per week unless the Minister grants an extension.")
CAND = {"text": SPAN, "has_table": False, "doc_id": "D", "topic": "t", "type": "legal", "start": 0, "end": len(SPAN),
        "block_ids": ["b1"], "n_tokens": 30}
def _stats():
    """문서 D는 "zephyr audit"를 많이 쓰고 다른 문서 둘(E, F)은 "board penalty"만 쓴다."""
    tf = {"D": Counter({"zephyr": 5, "audit": 4, "board": 3, "penalty": 2, "publish": 2}),
          "E": Counter({"board": 6, "penalty": 5, "minister": 4}), "F": Counter({"board": 4, "penalty": 3, "minister": 5})}
    return F.CorpusStats(tf)


STATS = _stats()


def gen(**kw):
    base = {"question": "How soon must the Zephyr Board publish its yearly audit after the year closes?",
            "answer": "Within 45 days of the financial year end.", "type": "extractive", "source": "text",
            "support": "must publish the annual audit within 45 days of the financial year end"}
    return {**base, **kw}


class Checks(unittest.TestCase):
    def reasons(self, **kw):
        return F.check_record(CAND, gen(**kw), STATS)[0]

    def test_good_record_passes(self):
        self.assertEqual(self.reasons(), [])

    def test_support_must_be_in_span_ignoring_whitespace(self):
        self.assertEqual(self.reasons(support="must  publish the\nannual audit within 45 days"), [])
        self.assertIn("support", self.reasons(support="must publish the monthly audit"))

    def test_answer_must_be_grounded(self):
        self.assertIn("grounding", self.reasons(answer="The board pays a bonus to directors every spring."))

    def test_deictic_question_is_rejected(self):
        self.assertIn("deictic", self.reasons(question="According to the passage, how soon must the Zephyr Board publish?"))

    def test_question_must_point_to_its_document(self):
        self.assertEqual(STATS.rank("zephyr audit", "D"), 1)
        old = F.SPECIFIC_MAX_RANK
        try:
            F.SPECIFIC_MAX_RANK = 1
            self.assertIn("specific", self.reasons(question="What penalty does the board impose on a minister?"))
            self.assertNotIn("specific", self.reasons())
        finally:
            F.SPECIFIC_MAX_RANK = old

    def test_copying_the_span_is_rejected(self):
        self.assertIn("copy", self.reasons(question="Zephyr: must publish the annual audit within 45 days of the financial year end?"))

    def test_evidence_quote_must_be_unique_in_the_document(self):
        quote = "must publish the annual audit within 45 days of the financial year end"
        once = F.CorpusStats(STATS.tf, {"D": F._norm(SPAN).lower()})
        twice = F.CorpusStats(STATS.tf, {"D": F._norm(SPAN + "\n\n" + SPAN).lower()})
        self.assertEqual(F.check_record(CAND, gen(support=quote), once)[0], [])
        self.assertIn("unique", F.check_record(CAND, gen(support=quote), twice)[0])

    def test_quote_in_another_document_also_counts_as_not_unique(self):
        quote = "must publish the annual audit within 45 days of the financial year end"
        other_edition = F.CorpusStats(STATS.tf, {"D": F._norm(SPAN).lower(), "E": F._norm("An older edition says that it " + quote).lower()})
        self.assertIn("unique", F.check_record(CAND, gen(support=quote), other_edition)[0])

    def test_copying_a_proper_name_is_allowed(self):
        cand = {**CAND, "text": "The Petroleum (Submerged Lands) (Management of Safety on Offshore Facilities) Regulations 2007 commence on a fixed day."}
        q = "When do the Petroleum (Submerged Lands) (Management of Safety on Offshore Facilities) Regulations 2007 take effect?"
        reasons, _ = F.check_record(cand, gen(question=q, answer="They commence on a fixed day.", support="commence on a fixed day"),
                                    F.CorpusStats({"D": Counter({"petroleum": 3, "regulations": 2}), "E": Counter({"board": 2})}))
        self.assertNotIn("copy", reasons)

    def test_table_source_requires_a_table(self):
        self.assertIn("source_without_table", self.reasons(source="table", type="table"))
        self.assertIn("schema", self.reasons(type="guess"))

    def test_duplicates_within_a_document_only(self):
        q = "How soon must the Zephyr Board publish its yearly audit?"
        self.assertEqual(F.find_duplicates([("a", "D", q), ("b", "D", q + " "), ("c", "E", q)]), {"b"})


class Assemble(unittest.TestCase):
    def test_vectara_shaped_meta_files(self):
        d = Path(tempfile.mkdtemp(dir=HERE))
        try:
            gens = {"D#0": gen(), "D#1": {"skip": "table of contents"}, "X#0": gen()}
            r = F.assemble({"D#0": CAND, "D#1": CAND}, gens, STATS, d)
            self.assertEqual((r["generated"], r["passed"], r["rejected"]), (3, 1, 2))
            q = json.loads((d / "queries.json").read_text(encoding="utf-8"))
            a = json.loads((d / "answers.json").read_text(encoding="utf-8"))
            qr = json.loads((d / "qrels.json").read_text(encoding="utf-8"))
            self.assertEqual(list(q), ["dq-0000"])
            self.assertEqual(set(q["dq-0000"]), {"query", "type", "source", "topic", "doc_type"})
            self.assertEqual(qr["dq-0000"]["doc_id"], "D")
            self.assertEqual(qr["dq-0000"]["end"], len(SPAN))
            self.assertIn("45 days", a["dq-0000"])
            self.assertEqual(sorted(json.loads((d / "rejected.json").read_text(encoding="utf-8"))), ["D#1", "X#0"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
