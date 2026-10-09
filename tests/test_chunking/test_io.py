"""chunking.io: 태그, 청크 dict·통계(옛 v1과 같은 결과), 내용 보존 검사, 파일 입출력(이어하기·force)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support
from support import FIX, count_words, runtime_for
from chunking import base, chunk_document
from chunking import io as cio
from chunking.v1 import meta as old_meta
from chunking.v1 import stats as old_stats
from common.docjson import load_doc, render


def fixture_chunks(key="A"):
    doc = load_doc(FIX / f"example_{key.lower()}.json")
    ctx = base.DocContext(doc, runtime_for(key))
    spans = chunk_document("semantic", ctx, p=70)
    return doc, ctx, spans


class TestTag(unittest.TestCase):
    def test_tags(self):
        self.assertEqual(cio.tag_for("A", {"p": 90}), "A__p90")
        self.assertEqual(cio.tag_for("A", {"p": 97.5}), "A__p97.5")
        self.assertEqual(cio.tag_for("fixed", {"T": 260}), "fixed__T260")
        self.assertEqual(cio.tag_for("fixed", {"size_chars": 500}), "fixed__c500")
        self.assertEqual(cio.tag_for("B", {"p": 90, "beta": 0.0}), "B__p90-beta0")
        self.assertEqual(cio.tag_for("E", {"p": 90, "delta_sigma": 2.0, "decay": 0.25}), "E__p90-decay0.25-delta_sigma2")
        self.assertEqual(cio.tag_for("header", {}), "header__none")
        self.assertEqual(cio.tag_for("header", {"T": 400}), "header__T400")
        self.assertEqual(cio.tag_for("A", {"p": 90, "beta": None}), "A__p90")          # None인 값은 이름에 넣지 않는다


class TestRecords(unittest.TestCase):
    def test_attach_meta_same_as_v1(self):
        for key in "AC":
            doc, ctx, spans = fixture_chunks(key)
            new = cio.attach_meta(doc, ctx.text, ctx.pieces, spans, lambda ts: [count_words(t) for t in ts], "semantic", {"p": 70})
            old = old_meta.attach_meta(doc, ctx.text, ctx.pieces, spans, count_words, "semantic", {"p": 70})
            self.assertEqual([{k: v for k, v in c.items() if k != "pages"} for c in new], old)      # pages는 새 필드, 나머지는 v1과 같다
            for c in new:
                self.assertEqual(c["text"], ctx.text[c["start"]:c["end"]])
                self.assertEqual((min(c["pages"]) if c["pages"] else None, max(c["pages"]) if c["pages"] else None),
                                 (c["page_start"], c["page_end"]))

    def test_pages_is_a_set_not_a_range(self):
        """각주가 문서 끝에 모여 있어도 쪽 번호는 블록의 원래 쪽을 유지한다: 끝쪽 청크의 pages에 사이 쪽이 끼지 않는다."""
        blocks = [{"id": "b0", "type": "paragraph", "text": "alpha beta gamma", "heading_path": [], "page": 9},
                  {"id": "b1", "type": "footnote", "text": "note one", "heading_path": [], "page": 3}]
        doc = {"doc_id": "d", "dataset": "ds", "fmt": "pdf", "lang": "en", "title": "t", "blocks": blocks}
        text, pieces = render(doc)
        chunks = cio.attach_meta(doc, text, pieces, [(0, len(text))], lambda ts: [count_words(t) for t in ts], "fixed", {})
        self.assertEqual(chunks[0]["pages"], [3, 9])
        self.assertEqual((chunks[0]["page_start"], chunks[0]["page_end"]), (3, 9))               # 범위는 사이 쪽(4~8)까지 부풀어 있다

    def test_stats_same_as_v1_on_shared_keys(self):
        doc, ctx, spans = fixture_chunks("A")
        chunks = cio.attach_meta(doc, ctx.text, ctx.pieces, spans, lambda ts: [count_words(t) for t in ts], "semantic", {})
        new = cio.compute_stats(chunks, {doc["doc_id"]: doc})
        old = old_stats.compute_stats(chunks, {doc["doc_id"]: doc})
        for k, v in old.items():
            self.assertEqual(new[k], v, k)
        self.assertIn("tokens_min", new)

    def test_coverage_ok(self):
        text = "abc def ghi"
        self.assertTrue(cio.coverage_ok(text, [(0, 3), (4, 7), (8, 11)]))
        self.assertFalse(cio.coverage_ok(text, [(0, 3), (8, 11)]))

    def test_chunk_documents_records_resolved_size(self):
        doc = load_doc(FIX / "example_a.json")
        recs, summary = cio.chunk_documents([doc], "fixed", {"T": 40}, runtime_for("A"), chars_per_token=4.0)
        self.assertEqual(summary["coverage_failures"], [])
        self.assertTrue(all(r["params"]["size_chars"] == 160 and r["params"]["lang"] == "en" for r in recs))
        self.assertEqual([r["chunk_id"] for r in recs][:2], [f"{doc['doc_id']}#0000", f"{doc['doc_id']}#0001"])
        text = render(doc)[0]
        self.assertTrue(all(r["text"] == text[r["start"]:r["end"]] for r in recs))

    def test_chunk_documents_all_chunkers_cover_text(self):
        doc = load_doc(FIX / "example_a.json")
        for name, params in (("fixed", {"size_chars": 160}), ("recursive", {"size_chars": 160}), ("header", {}), ("semantic", {"p": 70}),
                             ("A", {"p": 70}), ("B", {"p": 70}), ("E", {"p": 70})):
            recs, summary = cio.chunk_documents([doc], name, params, runtime_for("A"), chars_per_token=4.0)
            self.assertEqual(summary["coverage_failures"], [], name)
            self.assertTrue(recs, name)


class TestFiles(unittest.TestCase):
    def test_chunk_dataset_resume_and_force(self):
        doc = load_doc(FIX / "example_a.json")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "processed" / "testset").mkdir(parents=True)
            (d / "processed" / "testset" / f"{doc['doc_id']}.json").write_text(json.dumps({**doc, "dataset": "testset"}), encoding="utf-8")
            out = d / "chunks"
            with mock.patch.object(cio, "PROCESSED_DIR", d / "processed"), \
                    mock.patch("preprocess.datasets.load_entries", return_value=[{"doc_id": doc["doc_id"]}]):
                r1 = cio.chunk_dataset("testset", "semantic", {"p": 70}, runtime_for("A"), out_dir=out, log=lambda *a: None)
                self.assertFalse(r1["skipped"])
                path = out / "semantic__p70.jsonl"
                self.assertTrue(path.exists())
                rows = [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]
                self.assertEqual(len(rows), r1["n_chunks"])
                self.assertEqual(json.loads((out / "semantic__p70.meta.json").read_text(encoding="utf-8"))["params"], {"p": 70})
                r2 = cio.chunk_dataset("testset", "semantic", {"p": 70}, runtime_for("A"), out_dir=out, log=lambda *a: None)
                self.assertTrue(r2["skipped"])
                r3 = cio.chunk_dataset("testset", "semantic", {"p": 70}, runtime_for("A"), out_dir=out, force=True, log=lambda *a: None)
                self.assertFalse(r3["skipped"])
                self.assertFalse(list(out.glob("*.tmp")))


class TestRatio(unittest.TestCase):
    def test_measure_and_load(self):
        doc = load_doc(FIX / "example_a.json")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "processed" / "testset").mkdir(parents=True)
            (d / "processed" / "testset" / "a.json").write_text(json.dumps({**doc, "dataset": "testset"}), encoding="utf-8")
            with mock.patch.object(cio, "PROCESSED_DIR", d / "processed"), mock.patch.object(cio, "MANIFEST_DIR", d / "manifests"):
                res = cio.measure_ratio("testset", lambda ts: [len(t.split()) for t in ts])
                n_chars = len(render(doc)[0])
                n_words = len(render(doc)[0].split())
                self.assertAlmostEqual(res["by_lang"]["en"]["chars_per_token"], round(n_chars / n_words, 4))
                cio.save_ratio(res)
                self.assertAlmostEqual(cio.load_ratio("testset", "en"), round(n_chars / n_words, 4))
                with self.assertRaises(FileNotFoundError):
                    cio.load_ratio("nothing", "en")


if __name__ == "__main__":
    unittest.main()
