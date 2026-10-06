"""청킹 단위 테스트 - 가짜 임베딩(키워드 축)과 공백 단위 토큰 카운터를 쓴다(모델·GPU 불필요).
기대 경계는 기존 코드(legacy/src/chunkers.py)를 같은 입력·같은 설명용 파라미터로 한 번 돌려 얻은 값이다
(docs/restructure_plan.md 5.3절 예시 A·B와 새 예시 C). 실행: .venv311/bin/python -m unittest discover -s tests -v
"""
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from chunking import splitters  # noqa: E402
from chunking.meta import attach_meta  # noqa: E402
from chunking.smart import SmartChunker, cue_continues, split_sentences  # noqa: E402
from common.docjson import block_spans, load_doc, render, to_text  # noqa: E402

FIX = HERE / "fixtures"

AXES = {
    "A": ["free rid|honest|rational|agent|effort|participation", "reward|mechanism|truthful|contribution|align|monotone|strategy",
          "cifar|femnist|evaluate|simulated", "federated|train|model|data"],
    "B": ["cab|wheel|excuse|dopinder|nice|smells|lonesome|help", "wife|gita|cousin|bandhu|dishonorable|attractive|reason|called",
          "love|world|daffodil|yoga|mama|taste|bad|tight", "suit|christmas|pool"],
    "C": ["accuracy|benchmark|seeds|baseline", "cost|gpu|clients|round|hours", "privacy|encrypted|raw data|device|gradient"],
}
# 설명용 작은 파라미터: fixed 글자 수, strip, min/max(공백 단위 단어 수), smart mode
PARAMS = {"A": dict(fixed=160, strip=True, mn=25, mx=60, mode="pdf"),
          "B": dict(fixed=200, strip=False, mn=15, mx=40, mode="srt"),
          "C": dict(fixed=160, strip=True, mn=20, mx=45, mode="pdf")}
FILES = {"A": "example_a", "B": "example_b", "C": "example_c"}

EXPECTED = {
    "A": dict(fixed=[(0, 160), (160, 320), (320, 480), (480, 640), (640, 799), (800, 959), (960, 1069)],
              semantic=[(0, 16), (18, 388), (390, 401), (403, 583), (584, 937), (939, 1069)],
              smart=[(0, 16), (18, 388), (390, 602), (604, 937), (939, 1069)]),
    "B": dict(fixed=[(0, 200), (200, 400), (400, 600), (600, 691)],
              semantic=[(0, 138), (139, 354), (355, 394), (395, 611), (612, 690)],
              smart=[(0, 138), (139, 282), (283, 394), (395, 601), (602, 690)]),
    "C": dict(fixed=[(0, 160), (160, 320), (320, 480), (480, 586)],
              semantic=[(0, 283), (284, 408), (409, 586)],
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
    if chunker == "semantic":
        return {"percentile": 90, "min_tokens": p["mn"], "max_tokens": p["mx"]}
    return {"mode": p["mode"], "min_tokens": p["mn"], "max_tokens": p["mx"], "window": 2}


def run(key, chunker):
    doc = load_doc(FIX / f"{FILES[key]}.json")
    return doc, *splitters.chunk(doc, chunker, cfg_for(key, chunker), models_for(key))


class ExpectedBoundaries(unittest.TestCase):
    """예시 A(논문), B(자막), C(긴 문단)에서 세 청커의 청크 경계가 기존 코드의 결과와 같다."""

    def test_all(self):
        for key in "ABC":
            for chunker in ("fixed", "semantic", "smart"):
                with self.subTest(example=key, chunker=chunker):
                    _, chunks, _ = run(key, chunker)
                    self.assertEqual([(c["start"], c["end"]) for c in chunks], EXPECTED[key][chunker])

    def test_example_c_smart_cuts_one_paragraph_at_topic_changes(self):
        # 한 문단(9문장, 주제 3개)이 주제가 바뀌는 문장 경계에서만 갈린다
        doc, chunks, _ = run("C", "smart")
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
            self.assertEqual(info.get("semantic_dropped_chunks", 0), 0)


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


class SemanticParts(unittest.TestCase):
    def test_recover_spans_ignores_whitespace_and_keeps_order(self):
        flat = "A b\n\nC d\n\nA b"
        chunks = ["A b C d", "A b"]                       # 줄바꿈이 공백으로 바뀐 청크 + 앞에 나온 글이 반복됨
        spans = splitters.recover_spans(flat, chunks)
        self.assertEqual(spans, [(0, 8), (10, 13)])                                  # 두 번째는 뒤쪽 "A b"(앞쪽 0~3이 아님)
        self.assertEqual([flat[s:e] for s, e in spans], ["A b\n\nC d", "A b"])           # 줄바꿈이 살아 있는 원문 슬라이스

    def test_recover_spans_returns_none_when_not_found(self):
        self.assertEqual(splitters.recover_spans("hello world", ["zzz", ""]), [None, None])

    def test_unrecoverable_chunk_is_counted_not_silently_hidden(self):
        class FakeInner:
            def split_text(self, text):
                return ["# 1 Introduction", "text that is not in the document"]
        doc = load_doc(FIX / "example_a.json")
        text = to_text(doc)
        sp = splitters.SemanticSpanner(models_for("A")["embeddings"], count_words, 90, 25, 60, inner=FakeInner())
        spans, info = sp.spans(text)
        self.assertEqual(info["semantic_dropped_chunks"], 1)
        self.assertEqual(spans, [(0, 16)])

    def test_empty_text(self):
        sp = splitters.SemanticSpanner(models_for("A")["embeddings"], count_words, 90, 25, 60, inner=SimpleNamespace(split_text=lambda t: []))
        self.assertEqual(sp.spans("  \n "), ([], {"semantic_dropped_chunks": 0}))


class SmartParts(unittest.TestCase):
    def setUp(self):
        self.sm = SmartChunker(models_for("A")["embed_model"], count_words, mode="pdf", lang="en", min_tokens=25, max_tokens=60)

    def test_sections_split_at_headings(self):
        doc = load_doc(FIX / "example_a.json")
        text = to_text(doc)
        secs = self.sm.sections(text, block_spans(doc))
        self.assertEqual([[k for _, _, k in s] for s in secs],
                         [["heading", "para", "para"], ["heading", "para", "table", "para"], ["heading", "para"]])

    def test_long_paragraph_becomes_sentence_pieces_but_table_stays_whole(self):
        doc = load_doc(FIX / "example_c.json")
        text = to_text(doc)
        sm = SmartChunker(models_for("C")["embed_model"], count_words, mode="pdf", lang="en", min_tokens=20, max_tokens=45)
        kinds = [k for _, _, k in sm.sections(text, block_spans(doc))[0]]
        self.assertEqual(kinds, ["heading"] + ["sent"] * 9)
        docA = load_doc(FIX / "example_a.json")
        kindsA = [k for s in self.sm.sections(to_text(docA), block_spans(docA)) for _, _, k in s]
        self.assertEqual(kindsA.count("table"), 1)

    def test_split_sentences_offsets_are_literal(self):
        t = "One. Two? Three!"
        self.assertEqual([t[a:b] for a, b in split_sentences(t, 0, len(t))], ["One.", "Two?", "Three!"])

    def test_cue_continues_rules(self):
        self.assertTrue(cue_continues("Then the driver asked me if", "I wanted pizza.", "en"))     # 문장부호 없음
        self.assertTrue(cue_continues("Hello.", "and then", "en"))                                  # 다음 큐가 소문자로 시작
        self.assertTrue(cue_continues("Wait...", "Okay.", "en"))                                    # 말줄임
        self.assertTrue(cue_continues("Well,", "Okay.", "en"))                                      # 쉼표
        self.assertFalse(cue_continues("He left.", "She stayed.", "en"))
        self.assertTrue(cue_continues("그래서 우리는 가기로 했고", "다음날 떠났다.", "ko"))            # 연결어미
        self.assertFalse(cue_continues("그는 떠났다.", "그녀는 남았다.", "ko"))

    def test_continuation_bonus_moves_cut_away_from_a_split_sentence(self):
        # 큐 4 "Then the driver asked me if" → 큐 5 "I wanted pizza…" 사이는 주제도 바뀌지만 문장이 이어진다 (srt 모드)
        cues = ["The cab driver took the wheel.", "He said the cab was fast.", "The driver smiled at me.",
                "Then the driver asked me if", "I wanted pizza with cheese.", "The oven was already hot.",
                "Cheese melted in the oven.", "Pizza smelled great."]
        text = "\n".join(cues) + "\n"
        pieces, pos = [], 0
        for c in cues:
            pieces.append((pos, pos + len(c), "cue"))
            pos += len(c) + 1
        emb = make_stub(["cab|driver|wheel", "pizza|cheese|oven"])
        sm = SmartChunker(emb._client, count_words, mode="srt", lang="en", min_tokens=8, max_tokens=22, window=1)
        spans = sm.chunk_spans(text, pieces)
        grouped = [[i + 1 for i, (a, b, _) in enumerate(pieces) if a >= s and b <= e] for s, e in spans]
        self.assertEqual([(g[0], g[-1]) for g in grouped], [(1, 2), (3, 5), (6, 8)])               # 큐 4|5에서 자르지 않음


if __name__ == "__main__":
    unittest.main()
