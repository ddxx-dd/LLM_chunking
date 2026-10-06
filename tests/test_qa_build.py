"""QA 구축 단위 테스트 - 임시 폴더의 작은 원본을 쓴다(실제 data/ 는 건드리지 않는다)."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from support import HERE
from qa_build import build_qa as B


class Splits(unittest.TestCase):
    def test_deterministic_stratified_and_complete(self):
        strata = {f"d{i}": ("t" if i < 10 else "e") for i in range(15)}              # 층 t: 10개, e: 5개
        a, b = B.assign_splits(strata), B.assign_splits(strata)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(strata))
        self.assertEqual(sum(1 for d, s in a.items() if s == "tune" and strata[d] == "t"), 2)   # round(10*0.2)
        self.assertEqual(sum(1 for d, s in a.items() if s == "tune" and strata[d] == "e"), 1)   # round(5*0.2)

    def test_no_stratum_order_dependence(self):
        strata = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2}
        self.assertEqual(B.assign_splits(strata), B.assign_splits(dict(reversed(list(strata.items())))))


class Builders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=HERE))
        (self.tmp / "manifests").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_vectara(self):
        meta = self.tmp / "meta"
        (meta / "corpus").mkdir(parents=True)
        (meta / "queries.json").write_text(json.dumps({
            "q1": {"query": "what?", "type": "extractive", "source": "text"},
            "q2": {"query": "table?", "type": "abstractive", "source": "text-table"},
            "q3": {"query": "stray", "type": "extractive", "source": "text"}}), encoding="utf-8")
        (meta / "answers.json").write_text(json.dumps({"q1": "a1", "q2": "a2", "q3": "a3"}), encoding="utf-8")
        (meta / "qrels.json").write_text(json.dumps({"q1": {"doc_id": "P1", "section_id": 1}, "q2": {"doc_id": "P2", "section_id": 0},
                                                      "q3": {"doc_id": "NOPE", "section_id": 0}}), encoding="utf-8")
        for pid in ("P1", "P2"):
            (meta / "corpus" / f"{pid}.json").write_text(json.dumps({"sections": [
                {"section_id": "0", "text": f"{pid} intro"}, {"section_id": "1", "text": f"{pid} body"}]}), encoding="utf-8")
        (self.tmp / "manifests" / "vectara.json").write_text(json.dumps([{"paper_id": "P1"}, {"paper_id": "P2"}]), encoding="utf-8")
        with mock.patch.object(B, "MANIFEST_DIR", self.tmp / "manifests"):
            recs, problems = B.build_vectara(meta)
        self.assertEqual(problems, {"문서가 매니페스트에 없음": 1})
        by = {r["qid"]: r for r in recs}
        self.assertEqual(set(by), {"q1", "q2"})
        self.assertEqual(by["q1"]["evidence"], {"kind": "section_text", "section_id": 1, "text": "P1 body"})   # 정수 id도 문자열로 맞춤
        self.assertEqual((by["q2"]["qtype"], by["q2"]["source"]), ("abstractive", "text-table"))
        self.assertIn(by["q1"]["split"], ("tune", "eval"))
        self.assertFalse(by["q1"]["excluded"])

    def test_allganize_excludes_images_and_maps_files(self):
        meta = self.tmp / "meta"
        meta.mkdir()
        rows = [("finance", "x.pdf", "1", "table", "q-table"), ("finance", "x.pdf", "2", "image", "q-image"),
                ("law", "y.pdf", "", "paragraph", "q-nopage"), ("law", "gone.pdf", "3", "text", "q-orphan")]
        with open(meta / "rag_evaluation_result.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "question", "target_answer", "target_file_name", "target_page_no", "context_type"])
            for dom, name, page, ctx, q in rows:
                w.writerow([dom, q, "ans", name, page, ctx])
        (self.tmp / "manifests" / "allganize.json").write_text(json.dumps([
            {"doc_id": "X", "domain": "finance", "orig_file_name": "x.pdf"}, {"doc_id": "Y", "domain": "law", "orig_file_name": "y.pdf"}]), encoding="utf-8")
        with mock.patch.object(B, "MANIFEST_DIR", self.tmp / "manifests"):
            recs, problems = B.build_allganize(meta)
        self.assertEqual(problems, {"문서가 매니페스트에 없음": 1})
        by = {r["question"]: r for r in recs}
        self.assertEqual(by["q-table"]["doc_id"], "X")
        self.assertEqual(by["q-table"]["gold_page"], 1)
        self.assertTrue(by["q-image"]["excluded"])
        self.assertIsNone(by["q-image"]["split"])
        self.assertIsNone(by["q-nopage"]["gold_page"])
        self.assertIn(by["q-table"]["split"], ("tune", "eval"))
        self.assertEqual(by["q-table"]["split"], B.assign_splits({"X": "finance", "Y": "law"})["X"])  # 이미지 질문만 있는 문서는 층화에서 빠짐

    def test_write_and_load_filters(self):
        recs = [{"qid": "1", "doc_id": "A", "excluded": False, "split": "tune"}, {"qid": "2", "doc_id": "B", "excluded": False, "split": "eval"},
                {"qid": "3", "doc_id": "A", "excluded": True, "split": None}]
        with mock.patch.object(B, "QA_DIR", self.tmp / "qa"):
            B.write_qa("ds", recs)
            self.assertEqual([r["qid"] for r in B.load_qa("ds")], ["1", "2"])                    # 제외 질문은 기본으로 뺌
            self.assertEqual([r["qid"] for r in B.load_qa("ds", split="eval")], ["2"])
            self.assertEqual([r["qid"] for r in B.load_qa("ds", doc_ids=["A"])], ["1"])
            self.assertEqual(len(B.load_qa("ds", include_excluded=True)), 3)

    def test_load_qa_with_unicode_line_separators(self):
        recs = [{"qid": "1", "doc_id": "A", "excluded": False, "split": "eval", "question": "x\x85y\u2028z"},
                {"qid": "2", "doc_id": "A", "excluded": False, "split": "eval", "question": "plain"}]
        with mock.patch.object(B, "QA_DIR", self.tmp / "qa"):
            B.write_qa("ds", recs)
            got = B.load_qa("ds")
        self.assertEqual([r["qid"] for r in got], ["1", "2"])
        self.assertEqual(got[0]["question"], "x\x85y\u2028z")

    def test_unknown_dataset(self):
        with self.assertRaises(KeyError):
            B.build("docx_corpus")


if __name__ == "__main__":
    unittest.main()
