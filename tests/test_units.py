"""rag.units: 문서 목록과 읽기 (임시 폴더의 가짜 전처리 결과)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import units  # noqa: E402


class TestDocIds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        d = root / "allganize"
        d.mkdir()
        for name in ("b_doc", "_2019_report_like_doc", "a_doc"):          # '_'로 시작하는 진짜 문서도 있다
            (d / f"{name}.json").write_text(json.dumps({"doc_id": name, "blocks": []}), encoding="utf-8")
        (d / "_report.json").write_text("{}", encoding="utf-8")            # 실행 보고서는 문서가 아니다
        self.patch = mock.patch.object(units, "META_DIR", root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_report_excluded_underscore_docs_kept(self):
        self.assertEqual(units.doc_ids("allganize"), ["_2019_report_like_doc", "a_doc", "b_doc"])

    def test_load_doc(self):
        self.assertEqual(units.load_doc("allganize", "_2019_report_like_doc")["doc_id"], "_2019_report_like_doc")

    def test_unknown_dataset(self):
        with self.assertRaises(KeyError):
            units.doc_ids("nope")


if __name__ == "__main__":
    unittest.main()
