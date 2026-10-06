"""청크 통계·러너 단위 테스트 (가짜 임베딩, 실제 문서·모델 없음). 임시 파일은 tests/ 아래에 만들고 지운다."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from chunking import runner  # noqa: E402
from chunking.splitters import chunk  # noqa: E402
from chunking.stats import compute_stats, coverage_ok  # noqa: E402
from common.docjson import load_doc  # noqa: E402
from test_chunking import FILES, cfg_for, models_for  # noqa: E402

FIX = HERE / "fixtures"


def stats_for(key, chunker, **kw):
    doc = load_doc(FIX / f"{FILES[key]}.json")
    chunks, _ = chunk(doc, chunker, cfg_for(key, chunker), models_for(key))
    return compute_stats(chunks, {doc["doc_id"]: doc}, **kw), chunks


class Stats(unittest.TestCase):
    def test_pdf_example_a_structure_metrics(self):
        smart, _ = stats_for("A", "smart", min_tokens=25, max_tokens=60)
        semantic, _ = stats_for("A", "semantic", min_tokens=25, max_tokens=60)
        fixed, _ = stats_for("A", "fixed", min_tokens=25, max_tokens=60)
        self.assertEqual(smart["n_chunks"], 5)
        self.assertEqual((smart["table_split_rate"], semantic["table_split_rate"]), (0.0, 1.0))   # semantic은 표를 가름
        self.assertEqual(smart["partial_end_rate"], 0.0)                                          # smart는 블록 경계에서 끝남
        self.assertGreater(fixed["partial_end_rate"], 0.8)
        self.assertEqual(smart["sentence_cut_rate"], 0.0)
        self.assertGreater(fixed["sentence_cut_rate"], 0.5)                                       # fixed는 문장 중간에서 끝남
        self.assertAlmostEqual(smart["orphan_heading_rate"], 1 / 5)                               # 제목만 남은 청크 1개
        self.assertAlmostEqual(semantic["orphan_heading_rate"], 2 / 6, places=3)
        self.assertAlmostEqual(smart["under_min_rate"], 2 / 5)                                    # 3단어, 21단어 청크
        self.assertEqual(smart["over_max_rate"], 0.0)

    def test_srt_example_b_cues_split_ratio(self):
        fixed, _ = stats_for("B", "fixed")
        smart, _ = stats_for("B", "smart")
        semantic, _ = stats_for("B", "semantic")
        self.assertAlmostEqual(fixed["cues_split_ratio"], 3 / 19, places=3)        # 큐 9, 13, 16이 두 청크에 걸침
        self.assertEqual((smart["cues_split_ratio"], semantic["cues_split_ratio"]), (0.0, 0.0))
        self.assertNotIn("table_split_rate", smart)                      # srt에는 표·제목 지표가 없다

    def test_empty(self):
        self.assertEqual(compute_stats([], {}), {"n_chunks": 0})

    def test_coverage_ok(self):
        text = "ab cd\n\nef"
        self.assertTrue(coverage_ok(text, [(0, 5), (7, 9)]))             # 공백만 빠짐
        self.assertFalse(coverage_ok(text, [(0, 4), (7, 9)]))            # 'd' 손실
        self.assertFalse(coverage_ok(text, [(0, 9), (7, 9)]))            # 겹쳐서 중복


class TagsAndDocs(unittest.TestCase):
    def test_tag_for(self):
        self.assertEqual(runner.tag_for("fixed", {"target_tokens": 400, "overlap_ratio": 0.0}), "fixed__t400")
        self.assertEqual(runner.tag_for("fixed", {"target_tokens": 400, "overlap_ratio": 0.1, "strip_whitespace": False}), "fixed__t400-o10-nostrip")
        self.assertEqual(runner.tag_for("fixed", {"chunk_size_chars": 160}), "fixed__c160")
        self.assertEqual(runner.tag_for("semantic", {"percentile": 90, "min_tokens": 100, "max_tokens": 400}), "semantic__p90-min100-max400")
        self.assertEqual(runner.tag_for("smart", {"mode": "pdf", "min_tokens": 100, "max_tokens": 400, "window": 2}), "smart__min100-max400-w2")

    def test_chunk_documents_reports_no_failures_on_fixtures(self):
        docs = [load_doc(FIX / "example_a.json")]
        records, summary = runner.chunk_documents(docs, "smart", cfg_for("A", "smart"), models_for("A"))
        self.assertEqual(len(records), 5)
        self.assertEqual(summary, {"semantic_dropped_chunks": 0, "coverage_failures": []})

    def test_chunk_documents_flags_content_loss(self):
        docs = [load_doc(FIX / "example_a.json")]
        real = chunk

        def lossy(doc, chunker, cfg, models):               # 마지막 청크의 끝을 잘라 글자를 잃어버린 것처럼 만든다
            chunks, info = real(doc, chunker, cfg, models)
            chunks[-1] = {**chunks[-1], "end": chunks[-1]["end"] - 5}
            return chunks, info
        with mock.patch.object(runner, "chunk", lossy):
            _, summary = runner.chunk_documents(docs, "smart", cfg_for("A", "smart"), models_for("A"))
        self.assertEqual(summary["coverage_failures"], ["example_a"])


class ChunkDataset(unittest.TestCase):
    """chunk_dataset의 파일 쓰기·이어하기·force·meta (가짜 데이터셋 'fixture')."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=HERE))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, **kw):
        cfg = {"chunkers": {"smart": {"mode": "pdf", "min_tokens": 25, "max_tokens": 60, "window": 2}}}
        with mock.patch("preprocess.datasets.load_entries", return_value=[{"doc_id": "example_a"}]), \
                mock.patch.object(runner, "PROCESSED_DIR", FIX.parent / "fixtures_processed_tmp"), \
                mock.patch.object(runner, "dataset_cfg", return_value=cfg):
            return runner.chunk_dataset("fixture", "smart", models_for("A"), out_dir=self.tmp, log=lambda *_: None, **kw)

    def test_write_skip_force(self):
        proc = FIX.parent / "fixtures_processed_tmp" / "fixture"
        proc.mkdir(parents=True)
        try:
            shutil.copy(FIX / "example_a.json", proc / "example_a.json")
            r = self._run()
            out = self.tmp / "smart__min25-max60-w2.jsonl"
            self.assertFalse(r["skipped"])
            lines = [json.loads(x) for x in out.read_text(encoding="utf-8").split("\n") if x.strip()]
            self.assertEqual([(c["start"], c["end"]) for c in lines], [(0, 16), (18, 388), (390, 602), (604, 937), (939, 1069)])
            meta = json.loads((self.tmp / "smart__min25-max60-w2.meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["n_docs"], meta["n_chunks"], meta["semantic_dropped_chunks"], meta["coverage_failures"]), (1, 5, 0, []))
            self.assertFalse(list(self.tmp.glob("*.tmp")))                       # 임시 파일이 남지 않는다
            mtime = out.stat().st_mtime_ns
            self.assertTrue(self._run()["skipped"])                              # 이어하기: 건너뜀
            self.assertEqual(out.stat().st_mtime_ns, mtime)
            self.assertFalse(self._run(force=True)["skipped"])                   # force: 다시 만듦
        finally:
            shutil.rmtree(FIX.parent / "fixtures_processed_tmp")


if __name__ == "__main__":
    unittest.main()
