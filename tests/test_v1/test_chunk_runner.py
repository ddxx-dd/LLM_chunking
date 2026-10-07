"""청크 통계·러너 단위 테스트 (가짜 임베딩, 실제 문서·모델 없음). 임시 파일은 tests/ 아래에 만들고 지운다."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from support import HERE, FIX, FILES, models_for, cfg_for
from chunking.v1 import runner
from chunking.v1.splitters import chunk
from chunking.v1.stats import compute_stats, coverage_ok
from common.docjson import load_doc



def stats_for(key, chunker, **kw):
    doc = load_doc(FIX / f"{FILES[key]}.json")
    chunks, _ = chunk(doc, chunker, cfg_for(key, chunker), models_for(key))
    return compute_stats(chunks, {doc["doc_id"]: doc}, **kw), chunks


class Stats(unittest.TestCase):
    def test_pdf_example_a_structure_metrics(self):
        smart, _ = stats_for("A", "smart", min_tokens=25, max_tokens=60)
        semantic, _ = stats_for("A", "semantic", min_tokens=25, max_tokens=60)
        fixed, _ = stats_for("A", "fixed", min_tokens=25, max_tokens=60)
        self.assertEqual((smart["n_chunks"], semantic["n_chunks"]), (5, 4))
        self.assertEqual((smart["sentence_cut_rate"], semantic["sentence_cut_rate"]), (0.0, 0.0))  # 둘 다 문장 끝에서 끝남
        self.assertGreater(fixed["sentence_cut_rate"], 0.5)                                       # fixed는 문장 중간에서 끝남
        self.assertEqual(smart["orphan_heading_rate"], 0.0)                                       # smart: 제목으로 끝나는 청크 없음
        self.assertAlmostEqual(semantic["orphan_heading_rate"], 0.25)                             # semantic: "## 2 Method"가 청크 끝에 홀로 남음(4개 중 1개)
        self.assertEqual((smart["table_split_rate"], semantic["table_split_rate"]), (0.0, 0.0))   # 이 작은 예시에서는 둘 다 표를 안 가름
        self.assertAlmostEqual(smart["under_min_rate"], 0.2)                                      # 마지막 섹션(21단어)은 이웃 섹션과 합치면 max를 넘어 합칠 곳 없이 남음
        self.assertEqual(semantic["under_min_rate"], 0.0)                                         # semantic(섹션 분할 없음)은 DP가 min을 모두 지키는 분할을 찾는다
        self.assertEqual((smart["over_max_rate"], semantic["over_max_rate"]), (0.0, 0.0))

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
        self.assertEqual(runner.tag_for("smart", {"percentile": 90, "min_tokens": 100, "max_tokens": 400}), "smart__p90-min100-max400")
        self.assertEqual(runner.tag_for("smart", {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "buffer_size": 0}), "smart__p90-min100-max400")     # 기본(0)은 이름에 안 남김
        self.assertEqual(runner.tag_for("smart", {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "buffer_size": 1}), "smart__p90-min100-max400-b1")   # LangChain 기본
        only_table = {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "rules": {"table_atomic": True}}   # 어블레이션: 규칙 일부만
        self.assertEqual(runner.tag_for("smart", only_table), "smart__p90-min100-max400-rT")
        smart_default = {"heading_forced": False, "heading_glue": True, "table_atomic": True, "caption_glue": True, "para_atomic": False, "section_split": True}
        self.assertEqual(runner.tag_for("smart", {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "rules": smart_default}), "smart__p90-min100-max400")   # 기본 세트면 이름에 안 남김
        self.assertEqual(runner.tag_for("smart", {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "rules": {**smart_default, "heading_forced": True}}), "smart__p90-min100-max400-rHGTCS")
        self.assertEqual(runner.tag_for("semantic", {"percentile": 90, "min_tokens": 100, "max_tokens": 400, "rules": smart_default}), "semantic__p90-min100-max400-rGTCS")
        self.assertEqual(runner.tag_for("smart", {"mode": "srt", "algorithm": "legacy", "min_tokens": 128, "max_tokens": 512, "window": 2}), "smart__min128-max512-w2")

    def test_chunk_documents_reports_no_failures_on_fixtures(self):
        docs = [load_doc(FIX / "example_a.json")]
        records, summary = runner.chunk_documents(docs, "smart", cfg_for("A", "smart"), models_for("A"))
        self.assertEqual(len(records), 5)
        self.assertEqual(summary["coverage_failures"], [])
        self.assertEqual((summary["info_totals"]["smart_chunks"], summary["info_totals"]["smart_under_min"]), (5, 1))   # 청커가 센 통계가 합산됨

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
        cfg = {"chunkers": {"smart": {"percentile": 90, "min_tokens": 25, "max_tokens": 60}}}
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
            out = self.tmp / "smart__p90-min25-max60.jsonl"
            self.assertFalse(r["skipped"])
            lines = [json.loads(x) for x in out.read_text(encoding="utf-8").split("\n") if x.strip()]
            self.assertEqual([(c["start"], c["end"]) for c in lines], [(0, 195), (196, 388), (390, 602), (604, 937), (939, 1069)])
            meta = json.loads((self.tmp / "smart__p90-min25-max60.meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["n_docs"], meta["n_chunks"], meta["coverage_failures"]), (1, 5, []))
            self.assertEqual((meta["info_totals"]["smart_forced_cuts"], meta["info_totals"]["smart_under_min"]), (0, 1))     # 제목 앞 강제 절단은 기본 꺼짐; 마지막 섹션(21단어)은 합칠 곳 없어 min 미달로 남음
            self.assertFalse(list(self.tmp.glob("*.tmp")))                       # 임시 파일이 남지 않는다
            mtime = out.stat().st_mtime_ns
            self.assertTrue(self._run()["skipped"])                              # 이어하기: 건너뜀
            self.assertEqual(out.stat().st_mtime_ns, mtime)
            self.assertFalse(self._run(force=True)["skipped"])                   # force: 다시 만듦
        finally:
            shutil.rmtree(FIX.parent / "fixtures_processed_tmp")


if __name__ == "__main__":
    unittest.main()
