"""검색 지표·인덱스 단위 테스트 (가짜 인코더, 모델 불필요). 임시 파일은 tests/ 아래에 만들고 지운다."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from support import HERE
from indexing import index as I
from tasks import retrieval as R

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

    def test_page_hit_is_no_longer_measured(self):
        qa = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": "", "gold_page": 3}
        self.assertNotIn("page_hit", R.question_row(qa, [(CHUNKS[1], 1.0)], 3))      # allganize 쪽 지표는 측정하지 않는다(결정 2026-10-06)
        self.assertEqual(R.K_DEFAULT, 3)
        self.assertNotIn("allganize", R.PRIMARY)

    def test_span_coverage_counts_union_once_and_clips_to_the_evidence(self):
        ch = lambda s, e: {"start": s, "end": e}
        self.assertAlmostEqual(R.span_coverage([ch(0, 50)], 20, 60), 30 / 40)         # 근거 구간 밖은 세지 않는다
        self.assertAlmostEqual(R.span_coverage([ch(0, 40), ch(30, 60)], 20, 60), 1.0)  # 겹침은 한 번만(합집합)
        self.assertAlmostEqual(R.span_coverage([ch(0, 25), ch(40, 45)], 20, 60), 10 / 40)
        self.assertEqual(R.span_coverage([ch(0, 10), ch(70, 80)], 20, 60), 0.0)
        self.assertEqual(R.span_coverage([], 20, 60), 0.0)

    def test_evidence_recall_only_counts_gold_doc_chunks_and_aggregates(self):
        mk = lambda cid, doc, s, e: {**chunk(cid, doc, "t", 1, 1), "start": s, "end": e}
        qa = {"qid": "q", "doc_id": "A", "qtype": "t", "answer": "", "evidence": {"kind": "span", "start": 100, "end": 200}}
        hits = [(mk("a#0", "A", 0, 150), 1.0), (mk("b#0", "B", 0, 300), 0.9), (mk("a#1", "A", 150, 400), 0.8)]
        r = R.question_row(qa, hits, 3)
        self.assertAlmostEqual(r["evidence_recall"], 1.0)                                # 정답 문서의 두 청크가 구간을 덮음(다른 문서 청크는 제외)
        self.assertAlmostEqual(R.question_row(qa, hits[:1], 3)["evidence_recall"], 0.5)
        self.assertAlmostEqual(R.question_row(qa, hits[1:2], 3)["evidence_recall"], 0.0)  # 같은 위치라도 다른 문서면 0
        self.assertNotIn("evidence_recall", R.question_row({k: v for k, v in qa.items() if k != "evidence"}, hits, 3))
        agg = R.aggregate([r, R.question_row(qa, hits[:1], 3)])
        self.assertAlmostEqual(agg["evidence_recall@k"], 0.75)

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

    def test_combined_index_equals_an_index_built_from_all_chunks(self):
        x = [chunk("x#0", "X", "apple pie"), chunk("x#1", "X", "dog walks")]
        y = [chunk("y#0", "Y", "banana split"), chunk("y#1", "Y", "cherry apple tart")]
        ix, iy = I.build_index(x, fake_encode), I.build_index(y, fake_encode)
        comb = I.combine_indexes([ix, iy], fake_encode)
        whole = I.build_index(x + y, fake_encode)
        self.assertEqual([c["chunk_id"] for c in comb.chunks], ["x#0", "x#1", "y#0", "y#1"])
        for q in ("apple", "dog", "banana", "tart cherry"):
            a, b = comb.search([q], k=3)[0], whole.search([q], k=3)[0]
            self.assertEqual([c["chunk_id"] for c, _ in a], [c["chunk_id"] for c, _ in b])
            self.assertTrue(np.allclose([s for _, s in a], [s for _, s in b]))
        with self.assertRaises(ValueError):                                              # 청크 id가 겹치면 거부
            I.combine_indexes([ix, ix], fake_encode)

    def test_faiss_flat_equals_exact_numpy_search(self):
        # FAISS IndexFlatIP는 정확 검색이므로 이전 numpy 구현(기준)과 상위 k의 id·점수가 같아야 한다
        rng = np.random.default_rng(7)
        mat = rng.normal(size=(400, 48)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        qs = rng.normal(size=(25, 48)).astype(np.float32)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        chunks = [chunk(f"c#{i}", "D", f"t{i}") for i in range(400)]
        idx = I.ChunkIndex(chunks, mat, lambda texts: qs[[int(t) for t in texts]])
        got = idx.search([str(i) for i in range(25)], k=7)
        sims = qs @ mat.T
        for row, hits in zip(sims, got):
            order = np.argsort(-row, kind="stable")[:7]
            self.assertEqual([c["chunk_id"] for c, _ in hits], [f"c#{i}" for i in order])
            np.testing.assert_allclose([s for _, s in hits], row[order], rtol=1e-5, atol=1e-6)

    def test_k_larger_than_index_and_empty_index(self):
        idx = I.build_index(CHUNKS, fake_encode)
        self.assertEqual(len(idx.search(["apple"], k=50)[0]), len(CHUNKS))          # k가 청크 수보다 커도 있는 만큼만
        empty = I.build_index([], fake_encode)
        self.assertEqual(empty.search(["apple", "dog"], k=5), [[], []])
        self.assertEqual(empty.matrix.shape[0], 0)

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
            folder = self.tmp / "index" / "ds" / "tag"                                     # 저장 형식: FAISS 인덱스 + 청크 id + 정보
            self.assertTrue((folder / "index.faiss").exists())
            self.assertFalse((folder / "embeddings.npy").exists())                       # 예전 numpy 저장 형식은 더 쓰지 않는다
            info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
            self.assertEqual((info["backend"], info["n_chunks"], info["dim"]), ("faiss.IndexFlatIP", 4, 4))
            b = I.get_index("ds", "tag", self.encode)                                     # 캐시에서 불러옴 -> 인코딩 안 함
            self.assertEqual(self.calls, n_first)
            np.testing.assert_allclose(a.matrix, b.matrix)
            q = ["apple pie", "dog walks", "banana"]                                      # 파일에서 불러온 인덱스의 검색 = 처음 만든 것과 같다
            self.assertEqual([[(c["chunk_id"], round(s, 6)) for c, s in h] for h in b.search(q, 3)],
                             [[(c["chunk_id"], round(s, 6)) for c, s in h] for h in a.search(q, 3)])
            I.get_index("ds", "tag", self.encode, force=True)
            self.assertGreater(self.calls, n_first)                                       # force면 다시 만듦
            sub = I.get_index("ds", "tag", self.encode, doc_ids=["A"])                    # 표본(문서 일부) -> 청크 id가 달라 다시 만듦
            self.assertEqual([c["chunk_id"] for c in sub.chunks], ["a#0", "a#1"])
            self.assertEqual(json.loads((folder / "info.json").read_text(encoding="utf-8"))["n_chunks"], 2)   # 캐시가 표본 것으로 덮어써짐(섞이지 않음)

if __name__ == "__main__":
    unittest.main()
