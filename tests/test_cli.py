"""scripts/chunk.py, scripts/srt/chunk.py, scripts/run.py의 인자 처리와 규칙 (모델·GPU·데이터 없이)."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import support
from support import ROOT
import chunking
from chunking.embed import Runtime


@contextlib.contextmanager
def quiet():
    """argparse의 오류 문구(stderr)를 숨긴다(종료하는 것이 정상 동작인 테스트용)."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def load_script(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CH = load_script("cli_chunk", "scripts/chunk.py")
SRT = load_script("cli_srt_chunk", "scripts/srt/chunk.py")
RUN = load_script("cli_run", "scripts/run.py")


class TestChunkCli(unittest.TestCase):
    def parse(self, *argv):
        return CH.build_parser().parse_args(["--dataset", "vectara", *argv])

    def test_allowed_covers_every_chunker(self):
        self.assertEqual(set(CH.ALLOWED), set(chunking.CHUNKERS))

    def test_knob_required(self):
        with self.assertRaises(SystemExit):
            CH.params_for("A", self.parse("--chunker", "A"))
        with self.assertRaises(SystemExit):
            CH.params_for("fixed", self.parse("--chunker", "fixed"))
        self.assertEqual(CH.params_for("A", self.parse("--chunker", "A", "--p", "90")), {"p": 90.0})
        self.assertEqual(CH.params_for("fixed", self.parse("--chunker", "fixed", "--T", "260")), {"T": 260})
        self.assertEqual(CH.params_for("recursive", self.parse("--chunker", "recursive", "--size-chars", "500")), {"size_chars": 500})

    def test_header_without_value_is_pure(self):
        self.assertEqual(CH.params_for("header", self.parse("--chunker", "header")), {})
        self.assertEqual(CH.params_for("header", self.parse("--chunker", "header", "--T", "300")), {"T": 300})

    def test_only_allowed_values_pass(self):
        ns = self.parse("--chunker", "B", "--p", "90", "--beta", "0", "--delta", "0.3", "--T", "100")
        self.assertEqual(CH.params_for("B", ns), {"p": 90.0, "beta": 0.0})                # B는 delta·T를 받지 않는다
        ns = self.parse("--chunker", "E", "--p", "90", "--delta-sigma", "2", "--w", "9")
        self.assertEqual(CH.params_for("E", ns), {"p": 90.0, "delta_sigma": 2.0})

    def test_srt_dataset_rejected(self):
        with self.assertRaises(SystemExit), quiet():
            CH.build_parser().parse_args(["--dataset", "srt", "--chunker", "A", "--p", "90"])

    def test_missing_value_stops_before_loading_model(self):
        with mock.patch.object(Runtime, "load", side_effect=AssertionError("모델을 올리면 안 된다")):
            with self.assertRaises(SystemExit):
                CH.main(["--dataset", "vectara", "--chunker", "A"])

    def test_all_dispatches_each_chunker_with_its_own_knob(self):
        calls = []
        with mock.patch.object(Runtime, "load", return_value="RT") as load, \
                mock.patch("chunking.io.chunk_dataset", side_effect=lambda ds, name, params, rt, **kw: calls.append((ds, name, params, rt, kw))):
            CH.main(["--dataset", "docx_corpus", "--chunker", "all", "--p", "90", "--T", "260", "--limit", "2", "--force"])
        self.assertEqual({c[1] for c in calls}, set(chunking.CHUNKERS))
        by = {c[1]: c[2] for c in calls}
        self.assertEqual(by["A"], {"p": 90.0})
        self.assertEqual(by["fixed"], {"T": 260})
        self.assertEqual(by["header"], {"T": 260})
        self.assertTrue(all(c[3] == "RT" and c[4]["limit"] == 2 and c[4]["force"] for c in calls))
        self.assertTrue(load.call_args.kwargs["need_embeddings"])

    def test_textual_only_skips_embedding_model(self):
        with mock.patch.object(Runtime, "load", return_value="RT") as load, mock.patch("chunking.io.chunk_dataset"):
            CH.main(["--dataset", "vectara", "--chunker", "fixed", "--T", "260"])
        self.assertFalse(load.call_args.kwargs["need_embeddings"])


class TestSrtCli(unittest.TestCase):
    def test_overrides(self):
        ns = SRT.build_parser().parse_args(["--chunker", "semantic", "--percentile", "90", "--min-tokens", "128", "--max-tokens", "512"])
        self.assertEqual(SRT.overrides_for("semantic", ns), {"percentile": 90.0, "min_tokens": 128, "max_tokens": 512, "buffer_size": None})
        ns = SRT.build_parser().parse_args(["--chunker", "smart", "--window", "2"])
        self.assertEqual(SRT.overrides_for("smart", ns), {"min_tokens": None, "max_tokens": None, "window": 2})
        ns = SRT.build_parser().parse_args(["--chunker", "fixed", "--chunk-size-chars", "500", "--overlap", "0"])
        self.assertEqual(SRT.overrides_for("fixed", ns)["chunk_size_chars"], 500)


class TestRunCli(unittest.TestCase):
    def test_skip_existing_respects_precision(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "vectara" / "retrieval" / "A__p90.json"
            out.parent.mkdir(parents=True)
            self.assertFalse(RUN.skip_existing(out, False, "fp16"))                      # 없으면 실행
            out.write_text(json.dumps({"precision": "fp16"}), encoding="utf-8")
            self.assertTrue(RUN.skip_existing(out, False, "fp16"))                       # 같은 정밀도면 건너뜀
            self.assertFalse(RUN.skip_existing(out, True, "fp16"))                       # force면 다시
            out.write_text(json.dumps({}), encoding="utf-8")                              # precision 없음 = 옛 fp32 결과
            self.assertTrue(RUN.skip_existing(out, False, "fp16"))                       # 다른 정밀도라도 덮어쓰지 않는다(알림)

    def test_index_cache_name_by_precision(self):
        from indexing import index as I

        rows = [{"chunk_id": f"d#{i}", "doc_id": "d", "text": f"text {i}", "n_tokens": 2, "start": 0, "end": 1} for i in range(3)]
        calls = []

        def encode(texts):
            calls.append(len(texts))
            return np.eye(len(texts), 4, dtype=np.float32)

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "chunks" / "ds").mkdir(parents=True)
            (d / "chunks" / "ds" / "T.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with mock.patch.object(I, "CHUNKS_DIR", d / "chunks"), mock.patch.object(I, "INDEX_DIR", d / "index"):
                RUN.index_for("ds", "T", encode, ["d"], "fp16", False)
                self.assertTrue((d / "index" / "ds" / "T@fp16" / "index.faiss").exists())
                self.assertFalse((d / "index" / "ds" / "T").exists())
                n = len(calls)
                RUN.index_for("ds", "T", encode, ["d"], "fp16", False)                  # 캐시: 다시 임베딩하지 않는다
                self.assertEqual(len(calls), n)
                RUN.index_for("ds", "T", encode, ["d"], "fp32", False)                  # fp32는 옛 이름 그대로
                self.assertTrue((d / "index" / "ds" / "T" / "index.faiss").exists())

    def test_combined_mode_argument_checks(self):
        with self.assertRaises(SystemExit):
            RUN.main(["--task", "retrieval", "--datasets", "vectara", "--tag", "A__p90"])        # 1개는 합침이 아니다
        with self.assertRaises(SystemExit):
            RUN.main(["--task", "retrieval", "--datasets", "vectara,docx_corpus", "--tag", "all"])
        with self.assertRaises(SystemExit):
            RUN.main(["--task", "retrieval", "--datasets", "vectara,nope", "--tag", "A__p90"])
        with self.assertRaises(SystemExit), quiet():                                             # --dataset과 --datasets는 함께 못 쓴다
            RUN.build_parser().parse_args(["--task", "retrieval", "--dataset", "vectara", "--datasets", "a,b", "--tag", "x"])


class TestEndToEnd(unittest.TestCase):
    def test_new_chunk_file_works_with_index_and_evaluation(self):
        """chunking.io가 쓴 청크 파일을 indexing·tasks가 그대로 읽어 검색·채점까지 되는가(가짜 임베딩)."""
        from chunking import io as cio
        from common.docjson import load_doc
        from indexing import index as I
        from support import FIX, runtime_for
        from tasks.retrieval import evaluate

        doc = {**load_doc(FIX / "example_a.json"), "dataset": "ds"}
        rt = runtime_for("A")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            recs, summary = cio.chunk_documents([doc], "A", {"p": 70}, rt)
            self.assertEqual(summary["coverage_failures"], [])
            (d / "chunks" / "ds").mkdir(parents=True)
            (d / "chunks" / "ds" / "A__p70.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
            with mock.patch.object(I, "CHUNKS_DIR", d / "chunks"), mock.patch.object(I, "INDEX_DIR", d / "index"):
                idx = RUN.index_for("ds", "A__p70", rt.encode, [doc["doc_id"]], "fp16", False)
                qas = [{"qid": "q1", "dataset": "ds", "doc_id": doc["doc_id"], "qtype": "extractive", "evidence": {"kind": "none"},
                        "question": "free rid honest rational agent effort participation", "answer": "honest agents participate"}]
                res = evaluate(idx, qas, k=3)
        o = res["overall"]
        self.assertEqual(o["hit@k"], 1.0)
        self.assertGreater(o["mrr"], 0)
        self.assertEqual(res["index"]["n_chunks"], len(recs))


if __name__ == "__main__":
    unittest.main()
