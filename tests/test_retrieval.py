"""검색 지표·인덱스 단위 테스트 (가짜 인코더, 모델 불필요). 임시 파일은 tests/ 아래에 만들고 지운다."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from indexing import index as I  # noqa: E402
from tasks import retrieval as R  # noqa: E402

WORDS = ["apple", "banana", "cherry", "dog"]


def fake_encode(texts):
    """단어 4개의 개수 벡터(+작은 상수)를 정규화한 것 - 같은 단어를 가진 글끼리 유사하다."""
    out = []
    for t in texts:
        low = t.lower()
        v = np.array([low.count(w) for w in WORDS], dtype=np.float32) + 0.01
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


def chunk(cid, doc, text, p0=None, p1=None, n=10):
    return {"chunk_id": cid, "doc_id": doc, "text": text, "n_tokens": n, "page_start": p0, "page_end": p1}


CHUNKS = [chunk("a#0", "A", "apple apple apple pie", 1, 1), chunk("a#1", "A", "dog dog walks", 2, 3),
          chunk("b#0", "B", "banana banana split", 1, 1), chunk("c#0", "C", "cherry cherry tart", 5, 5)]


class Metrics(unittest.TestCase):
    def test_f1_hand_computed(self):
        self.assertAlmostEqual(R.f1_score("the cat sat", "the cat"), 0.8)          # p=1, r=2/3
        self.assertEqual(R.f1_score("x y", "z"), 0.0)
        self.assertEqual(R.f1_score("", "z"), 0.0)
        self.assertAlmostEqual(R.f1_score("The Cat!", "the cat"), 1.0)             # 정규화

    def test_containment(self):
        self.assertAlmostEqual(R.containment("a b c d", "a b x", n=1), 0.5)
        self.assertAlmostEqual(R.containment("a b c d", "a b x", n=2), 1 / 3)       # (a,b)만 포함, 2-gram 3개
        self.assertEqual(R.containment("", "a b"), 0.0)

    def test_question_row_hit_rank_and_mrr(self):
        qa = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": "dog walks"}
        hits = [(CHUNKS[2], .9), (CHUNKS[3], .8), (CHUNKS[1], .7)]                  # 정답 문서 청크가 3위
        r = R.question_row(qa, hits, k=5)
        self.assertEqual((r["hit"], r["rank"]), (True, 3))
        self.assertAlmostEqual(r["mrr"], 1 / 3)
        self.assertAlmostEqual(r["f1"], R.f1_score("dog walks", "dog dog walks"))
        self.assertEqual(r["top_k_tokens"], 30)
        miss = R.question_row(qa, hits[:2], k=5)
        self.assertEqual((miss["hit"], miss["mrr"], miss["f1"]), (False, 0.0, 0.0))
        self.assertFalse(R.question_row(qa, hits, k=2)["hit"])                      # k=2면 3위는 밖

    def test_section_hit_only_counts_gold_doc_chunks(self):
        qa = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": "", "evidence": {"kind": "section_text", "text": "apple pie recipe"}}
        self.assertTrue(R.question_row(qa, [(CHUNKS[0], 1.0)], 5)["section_hit"])    # apple/pie 포함 -> 2/4 = 0.5 이상
        other_doc = chunk("x#0", "B", "apple apple apple pie", 1, 1)
        self.assertFalse(R.question_row(qa, [(other_doc, 1.0)], 5)["section_hit"])  # 내용이 같아도 다른 문서면 0
        self.assertFalse(R.question_row(qa, [(CHUNKS[1], 1.0)], 5)["section_hit"])  # 같은 문서의 다른 섹션 청크
        no_ev = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": ""}
        self.assertNotIn("section_hit", R.question_row(no_ev, [(CHUNKS[0], 1.0)], 5))

    def test_page_hit(self):
        qa = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": "", "gold_page": 3}
        self.assertTrue(R.question_row(qa, [(CHUNKS[1], 1.0)], 5)["page_hit"])      # 2~3쪽 청크가 3쪽을 포함
        self.assertFalse(R.question_row(qa, [(CHUNKS[0], 1.0)], 5)["page_hit"])     # 1쪽
        self.assertFalse(R.question_row({**qa, "gold_page": 1}, [(chunk("n", "A", "t"), 1.0)], 5)["page_hit"])   # 쪽 정보 없는 청크
        self.assertNotIn("page_hit", R.question_row({**qa, "gold_page": None}, [(CHUNKS[1], 1.0)], 5))

    def test_aggregate_and_by_qtype(self):
        idx = I.ChunkIndex(CHUNKS, fake_encode([c["text"] for c in CHUNKS]), fake_encode)
        qas = [{"qid": "1", "doc_id": "A", "qtype": "x", "question": "apple pie", "answer": "apple pie"},
               {"qid": "2", "doc_id": "B", "qtype": "y", "question": "banana", "answer": "banana split"},
               {"qid": "3", "doc_id": "C", "qtype": "y", "question": "dog", "answer": "cherry tart"}]       # 질문이 dog -> 다른 문서
        res = R.evaluate(idx, qas, k=1)
        self.assertEqual([r["hit"] for r in res["rows"]], [True, True, False])
        self.assertAlmostEqual(res["overall"]["hit@k"], 2 / 3)
        self.assertAlmostEqual(res["by_qtype"]["y"]["hit@k"], 0.5)
        self.assertEqual(res["by_qtype"]["x"]["n"], 1)
        self.assertEqual(res["index"]["n_chunks"], 4)
        self.assertEqual(R.aggregate([]), None)


class Index(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=HERE))
        self.calls = 0

        def counting(texts):
            self.calls += 1
            return fake_encode(texts)
        self.encode = counting

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_search_order_and_stable_ties(self):
        idx = I.build_index(CHUNKS, fake_encode)
        top = idx.search(["apple"], k=2)[0]
        self.assertEqual(top[0][0]["chunk_id"], "a#0")
        same = [chunk("t#0", "A", "banana"), chunk("t#1", "A", "banana")]
        tie = I.build_index(same, fake_encode).search(["banana"], k=2)[0]
        self.assertEqual([c["chunk_id"] for c, _ in tie], ["t#0", "t#1"])              # 동점은 청크 순서

    def test_chunk_file_with_unicode_line_separators_loads(self):
        # 한국어 PDF 원문에는 U+0085(NEL)가 섞여 있다. str.splitlines()는 여기서 줄을 나눠 JSON을 깨뜨렸다(회귀 방지)
        tricky = [chunk("n#0", "A", "앞\x85뒤 and\u2028more", 1, 1), chunk("n#1", "A", "plain", 2, 2)]
        d = self.tmp / "chunks" / "ds"
        d.mkdir(parents=True)
        (d / "t.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in tricky) + "\n", encoding="utf-8")
        with mock.patch.object(I, "CHUNKS_DIR", self.tmp / "chunks"):
            got = I.load_chunks("ds", "t")
        self.assertEqual([c["chunk_id"] for c in got], ["n#0", "n#1"])
        self.assertEqual(got[0]["text"], "앞\x85뒤 and\u2028more")

    def test_truncation_count_recorded(self):
        big = CHUNKS + [chunk("z#0", "Z", "dog", n=9000)]
        self.assertEqual(I.build_index(big, fake_encode, max_seq_length=8192).truncated, 1)

    def test_save_load_cache_and_invalidation(self):
        chunks_dir = self.tmp / "chunks"
        (chunks_dir / "ds").mkdir(parents=True)
        (chunks_dir / "ds" / "tag.jsonl").write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
        with mock.patch.object(I, "CHUNKS_DIR", chunks_dir), mock.patch.object(I, "INDEX_DIR", self.tmp / "index"):
            a = I.get_index("ds", "tag", self.encode)
            n_first = self.calls
            b = I.get_index("ds", "tag", self.encode)                                     # 캐시에서 불러옴 -> 인코딩 안 함
            self.assertEqual(self.calls, n_first)
            np.testing.assert_allclose(a.matrix, b.matrix)
            I.get_index("ds", "tag", self.encode, force=True)
            self.assertGreater(self.calls, n_first)                                       # force면 다시 만듦
            sub = I.get_index("ds", "tag", self.encode, doc_ids=["A"])                    # 표본(문서 일부) -> 청크 id가 달라 다시 만듦
            self.assertEqual([c["chunk_id"] for c in sub.chunks], ["a#0", "a#1"])
            self.assertTrue((self.tmp / "index" / "ds" / "tag" / "info.json").exists())


if __name__ == "__main__":
    unittest.main()
