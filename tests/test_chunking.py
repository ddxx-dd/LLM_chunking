"""청킹 공통 테스트(글자수 분할, 청커 등록부, 불변식, 메타데이터) - 가짜 임베딩(키워드 축)과 공백 단위 토큰 카운터를 쓴다.
fixed의 기대 경계는 기존 코드(legacy/src/chunkers.py)와 같고, semantic·smart는 공유 엔진(semantic_core)의 결과를
손으로 검토해 고정한 값이다(예시 A=논문, B=자막, C=긴 문단; docs/restructure_plan.md 5.3절).
엔진·규칙 테스트는 test_semantic_engine.py / test_constraints.py, 보관용 legacy는 test_legacy_smart.py.
실행: .venv311/bin/python -m unittest discover -s tests -v
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from chunking import splitters  # noqa: E402
from chunking.meta import attach_meta  # noqa: E402
from common.docjson import load_doc, render, to_text  # noqa: E402

FIX = HERE / "fixtures"

AXES = {
    "A": ["free rid|honest|rational|agent|effort|participation", "reward|mechanism|truthful|contribution|align|monotone|strategy",
          "cifar|femnist|evaluate|simulated", "federated|train|model|data"],
    "B": ["cab|wheel|excuse|dopinder|nice|smells|lonesome|help", "wife|gita|cousin|bandhu|dishonorable|attractive|reason|called",
          "love|world|daffodil|yoga|mama|taste|bad|tight", "suit|christmas|pool"],
    "C": ["accuracy|benchmark|seeds|baseline", "cost|gpu|clients|round|hours", "privacy|encrypted|raw data|device|gradient"],
}
# 설명용 작은 파라미터: fixed 글자 수, strip, min/max(공백 단위 단어 수), mode(srt는 legacy smart 설정이 쓴다)
PARAMS = {"A": dict(fixed=160, strip=True, mn=25, mx=60, mode="pdf"),
          "B": dict(fixed=200, strip=False, mn=15, mx=40, mode="srt"),
          "C": dict(fixed=160, strip=True, mn=20, mx=45, mode="pdf")}
FILES = {"A": "example_a", "B": "example_b", "C": "example_c"}

EXPECTED = {
    # 거리 계산은 buffer_size=0(단위 하나씩)이 기본이다.
    # A: smart는 모든 "##" 제목이 청크의 시작이고(390, 939), 표는 제목 + 문단과 한 청크(390~602)에 통째로 들어간다.
    #    마지막 청크(939~, 21단어)는 min 25 미달인데 앞 청크(49단어)와 합치면 max 60을 넘어 합치지 못하고 위반으로 남는다.
    #    semantic은 제목을 모르므로 "## 2 Method"가 청크 끝(196~401)에 홀로 남는다.
    "A": dict(fixed=[(0, 160), (160, 320), (320, 480), (480, 640), (640, 799), (800, 959), (960, 1069)],
              semantic=[(0, 195), (196, 401), (403, 602), (604, 780), (781, 1069)],
              smart=[(0, 195), (196, 388), (390, 602), (604, 937), (939, 1069)]),
    # B(srt): smart는 아직 legacy 설정(큐 단위 경계 점수), semantic은 공유 엔진
    "B": dict(fixed=[(0, 200), (200, 400), (400, 600), (600, 691)],
              semantic=[(0, 138), (139, 282), (283, 394), (395, 497), (498, 690)],
              smart=[(0, 138), (139, 282), (283, 394), (395, 601), (602, 690)]),
    # C: 제목 + 한 문단(주제 3개). 표·캡션이 없어서 semantic과 smart가 같다. 컷이 주제 전환 문장 경계에 정확히 생긴다.
    "C": dict(fixed=[(0, 160), (160, 320), (320, 480), (480, 586)],
              semantic=[(0, 212), (213, 408), (409, 586)],
              smart=[(0, 212), (213, 408), (409, 586)]),
}


def make_stub(axes):
    def vec(t):
        tl = t.lower()
        v = np.array([len(re.findall(a, tl)) for a in axes], float) + 0.05
        return v / np.linalg.norm(v)

    class Emb:
        def embed_documents(self, ts):
            return [vec(t).tolist() for t in ts]

        def embed_query(self, t):
            return vec(t).tolist()

        @property
        def _client(self):
            class Client:
                def encode(s, ts, normalize_embeddings=True):
                    return np.array([vec(t) for t in ts])
            return Client()
    return Emb()


def count_words(s):
    return len(s.split())


def models_for(key):
    emb = make_stub(AXES[key])
    return {"embeddings": emb, "embed_model": emb._client, "count": count_words}


def cfg_for(key, chunker):
    p = PARAMS[key]
    if chunker == "fixed":
        return {"chunk_size_chars": p["fixed"], "strip_whitespace": p["strip"], "overlap_ratio": 0.0}
    if chunker == "smart" and p["mode"] == "srt":                      # srt smart는 아직 legacy
        return {"mode": "srt", "algorithm": "legacy", "min_tokens": p["mn"], "max_tokens": p["mx"], "window": 2}
    return {"percentile": 90, "min_tokens": p["mn"], "max_tokens": p["mx"]}     # semantic(규칙 off) / smart(규칙 on)


def run(key, chunker):
    doc = load_doc(FIX / f"{FILES[key]}.json")
    return doc, *splitters.chunk(doc, chunker, cfg_for(key, chunker), models_for(key))


class ExpectedBoundaries(unittest.TestCase):
    """예시 A(논문), B(자막), C(긴 문단)에서 세 청커의 청크 경계가 고정해 둔 값과 같다."""

    def test_all(self):
        for key in "ABC":
            for chunker in ("fixed", "semantic", "smart"):
                with self.subTest(example=key, chunker=chunker):
                    _, chunks, _ = run(key, chunker)
                    self.assertEqual([(c["start"], c["end"]) for c in chunks], EXPECTED[key][chunker])

    def test_example_c_cuts_fall_exactly_on_the_topic_changes(self):
        # 한 문단(9문장, 주제 3개). buffer 0에서는 주제가 바뀌는 문장 경계에서 정확히 잘린다(semantic도 같다).
        for chunker in ("semantic", "smart"):
            doc, chunks, _ = run("C", chunker)
            text = to_text(doc)
            self.assertEqual([text.startswith(p, c["start"]) for c, p in zip(chunks, ["# 3 Results", "Training cost", "Privacy is"])],
                             [True, True, True])
            self.assertTrue(all(c["text"].rstrip().endswith(".") for c in chunks))     # 문장 끝에서만 끝남


class Invariants(unittest.TestCase):
    """docs/restructure_plan.md 12절 3단계 테스트 ①~⑤ + 내용 보존."""

    def each(self):
        for key in "ABC":
            for chunker in ("fixed", "semantic", "smart"):
                yield key, chunker, *run(key, chunker)

    def test_1_chunk_text_is_literal_slice(self):
        for key, chunker, doc, chunks, _ in self.each():
            text = to_text(doc)
            for c in chunks:
                self.assertEqual(c["text"], text[c["start"]:c["end"]], (key, chunker))

    def test_2_ordered_and_non_overlapping(self):
        for key, chunker, doc, chunks, _ in self.each():
            for a, b in zip(chunks, chunks[1:]):
                self.assertLessEqual(a["end"], b["start"], (key, chunker))

    def test_3_metadata_matches_offsets(self):
        for key, chunker, doc, chunks, _ in self.each():
            text, pieces = render(doc)
            for c in chunks:
                hit = [p for p in pieces if p.end > c["start"] and p.start < c["end"]]
                self.assertEqual(c["block_ids"], [b for p in hit for b in p.block_ids if b], (key, chunker))
                self.assertEqual(c["partial_start"], bool(hit) and c["start"] > hit[0].start)
                self.assertEqual(c["partial_end"], bool(hit) and c["end"] < hit[-1].end)

    def test_5_deterministic(self):
        for key in "ABC":
            for chunker in ("fixed", "semantic", "smart"):
                self.assertEqual(run(key, chunker)[1], run(key, chunker)[1], (key, chunker))

    def test_6_no_content_lost(self):
        # 청크들의 공백 아닌 글자를 이어 붙이면 원문의 공백 아닌 글자와 같다 (overlap 0)
        for key, chunker, doc, chunks, info in self.each():
            text = to_text(doc)
            kept = re.sub(r"\s+", "", "".join(c["text"] for c in chunks))
            self.assertEqual(kept, re.sub(r"\s+", "", text), (key, chunker))


class Metadata(unittest.TestCase):
    def test_srt_smart_chunk_has_cue_times_and_whole_cues(self):
        doc, chunks, _ = run("B", "smart")
        c = chunks[0]                                   # 큐 1~7
        self.assertEqual(c["block_ids"], [f"c{i}" for i in range(1, 8)])
        self.assertFalse(c["partial_start"] or c["partial_end"])
        self.assertAlmostEqual(c["t_start"], 180.430, places=3)
        self.assertAlmostEqual(c["t_end"], 202.905, places=3)
        self.assertTrue(all(not c["partial_start"] and not c["partial_end"] for c in chunks))   # smart는 큐를 가르지 않는다

    def test_srt_fixed_splits_a_cue(self):
        doc, chunks, _ = run("B", "fixed")
        self.assertTrue(chunks[0]["partial_end"] and chunks[1]["partial_start"])
        self.assertEqual(chunks[0]["block_ids"][-1], "c9")
        self.assertEqual(chunks[1]["block_ids"][0], "c9")                      # 같은 큐가 두 청크에 걸침
        self.assertAlmostEqual(chunks[1]["t_start"], 213.380, places=3)

    def test_pdf_smart_chunk_has_blocks_page_and_heading_path(self):
        doc, chunks, _ = run("A", "smart")
        c = chunks[1]                                   # (18, 388): 문단 두 개
        self.assertEqual(c["block_ids"], ["b1", "b2"])
        self.assertEqual((c["page_start"], c["page_end"]), (1, 1))
        self.assertEqual(c["heading_path"], ["1 Introduction"])
        self.assertIsNone(c["t_start"])
        self.assertEqual(c["chunk_id"], "example_a#0001")
        self.assertEqual(c["n_tokens"], count_words(c["text"]))

    def test_chunk_in_gap_between_pieces_has_no_blocks(self):
        doc = load_doc(FIX / "example_a.json")
        text, pieces = render(doc)
        gap = (pieces[0].end, pieces[1].start)           # 제목과 문단 사이 "\n\n"
        rec = attach_meta(doc, text, pieces, [gap], count_words, "x", {})[0]
        self.assertEqual((rec["block_ids"], rec["partial_start"], rec["partial_end"]), ([], False, False))


class FixedOverlap(unittest.TestCase):
    def test_overlap_keeps_slice_invariant_and_overlap_size(self):
        doc = load_doc(FIX / "example_a.json")
        cfg = {"chunk_size_chars": 200, "strip_whitespace": True, "overlap_ratio": 0.2}
        chunks, _ = splitters.chunk(doc, "fixed", cfg, models_for("A"))
        text = to_text(doc)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertEqual(c["text"], text[c["start"]:c["end"]])
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a["end"] - b["start"], 40)                    # 겹침은 설정값(200×0.2) 이내
        self.assertEqual(chunks[0]["params"]["chunk_overlap_chars"], 40)

    def test_resolve_fixed_size_from_ratio_file(self):
        ratio = HERE.parent / "data" / "manifests" / "vectara.ratio.json"
        if not ratio.exists():
            self.skipTest("vectara.ratio.json 없음")
        size, ov = splitters.resolve_fixed_size({"dataset": "vectara", "lang": "en"}, {"target_tokens": 400, "overlap_ratio": 0.1})
        self.assertAlmostEqual(size, 1343, delta=10)
        self.assertEqual(ov, round(0.1 * size))


if __name__ == "__main__":
    unittest.main()
