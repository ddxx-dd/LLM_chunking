"""제약 기반 의미 분할(ConstrainedSmartChunker) 단위 테스트 - 가짜 임베딩(키워드 축)과 공백 단위 토큰 카운터.
5개 제약: ① 제목 앞 강제 경계 ② 제목 뒤 금지 ③ 표 분할 금지 ④ 표-캡션 금지 ⑤ 문단 경계 선호."""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from chunking import splitters  # noqa: E402
from chunking.smart import ConstrainedSmartChunker, SmartChunker, caption_glue  # noqa: E402
from common.docjson import load_doc, render, to_text  # noqa: E402
from test_chunking import count_words  # noqa: E402

TOPICS = ["alpha", "beta", "gamma"]


class Emb:
    """주제어(alpha/beta/gamma) 개수로 만든 벡터 - 같은 주제 글끼리 비슷하다."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in TOPICS], float) + 0.05
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def words(topic, n):
    return " ".join([topic] * n) + "."


def build(parts, sep="\n\n"):
    """[(kind, text)] -> (평문, [(start, end, kind)])"""
    text, blocks = "", []
    for kind, t in parts:
        if text:
            text += sep
        s = len(text)
        text += t
        blocks.append((s, s + len(t), kind))
    return text, blocks


def run(parts, mn=20, mx=100, pct=90, glue=(), **kw):
    text, blocks = build(parts)
    ch = ConstrainedSmartChunker(Emb(), count_words, "pdf", "en", mn, mx, window=kw.pop("window", 2), percentile=pct, **kw)
    return text, blocks, ch.chunk_spans(text, blocks, glue), ch


class Constraint1_HeadingStartsChunk(unittest.TestCase):
    def test_every_section_start_is_a_chunk_start(self):
        parts = [("heading", "# One"), ("para", words("alpha", 30)), ("heading", "# Two"), ("para", words("beta", 30)),
                 ("heading", "# Three"), ("para", words("gamma", 30))]
        text, blocks, spans, ch = run(parts)
        self.assertEqual([s for s, _ in spans], [blocks[0][0], blocks[2][0], blocks[4][0]])
        self.assertEqual(ch.stats["forced_cuts"], 2)
        self.assertEqual((ch.stats["under_min"], ch.stats["over_max"]), (0, 0))

    def test_short_section_is_merged_forward_only_because_it_is_below_min(self):
        parts = [("heading", "# A"), ("para", words("alpha", 3)), ("heading", "# B"), ("para", words("beta", 3)),
                 ("heading", "# C"), ("para", words("gamma", 3)), ("heading", "# D"), ("para", words("alpha", 30))]
        text, blocks, spans, ch = run(parts, mn=10, mx=100)
        sizes = [count_words(text[s:e]) for s, e in spans]
        self.assertTrue(all(10 <= x <= 100 for x in sizes), sizes)
        self.assertEqual(ch.stats["merges"], 2)
        self.assertEqual(spans[0][0], blocks[0][0])


class Constraint2_NoCutAfterHeading(unittest.TestCase):
    def test_consecutive_headings_stay_together(self):
        parts = [("heading", "# 1 Intro"), ("heading", "## 1.1 Scope"), ("para", words("alpha", 30)),
                 ("heading", "# 2 Method"), ("para", words("beta", 30))]
        text, blocks, spans, _ = run(parts)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], (blocks[0][0], blocks[2][1]))        # 제목 두 개 + 문단이 한 청크

    def test_no_chunk_ends_with_a_heading_even_when_size_split_is_needed(self):
        parts = [("heading", "# Big"), ("para", words("alpha", 40)), ("heading", "## Sub"), ("para", words("alpha", 40)),
                 ("heading", "## Sub2"), ("para", words("alpha", 40))]
        text, blocks, spans, _ = run(parts, mn=20, mx=60)
        heading_ends = {e for (s, e, k) in blocks if k == "heading"}
        self.assertFalse(heading_ends & {e for _, e in spans})


class Constraint3_TableIsNeverSplit(unittest.TestCase):
    def test_oversized_table_is_one_chunk_and_nothing_cuts_through_it(self):
        parts = [("heading", "# T"), ("para", words("alpha", 20)), ("table", words("beta", 80)), ("para", words("gamma", 20))]
        text, blocks, spans, ch = run(parts, mn=10, mx=50)
        table = (blocks[2][0], blocks[2][1])
        self.assertIn(table, spans)                                       # 표 하나가 통째로 한 청크
        for s, e in spans:
            self.assertFalse(s < table[1] and e > table[0] and (s, e) != table and not (s <= table[0] and e >= table[1]))
        self.assertEqual(ch.stats["over_max"], 0)                         # 표는 위반으로 세지 않는다


class Constraint4_CaptionStaysWithTable(unittest.TestCase):
    def parts(self):
        # 본문(beta)과 캡션(beta)은 같은 주제, 표(gamma)만 다르다 -> 제약이 없으면 캡션|표 경계가 가장 자르기 좋다
        return [("heading", "# C"), ("para", words("beta", 25)), ("para", words("beta", 8)), ("table", words("gamma", 25)),
                ("para", words("beta", 25)), ("para", words("beta", 25)), ("para", words("beta", 25)), ("para", words("beta", 25))]

    def test_without_glue_the_cut_falls_between_caption_and_table(self):
        text, blocks, spans, _ = run(self.parts(), mn=10, mx=300, pct=60, window=1)   # window 1: 표 앞뒤 거리가 대칭
        self.assertIn(blocks[3][0], [s for s, _ in spans])                # 표가 청크 시작 = 캡션|표 사이를 잘랐다

    def test_with_glue_the_cut_is_not_made_there(self):
        blocks = build(self.parts())[1]
        text, blocks, spans, _ = run(self.parts(), mn=10, mx=300, pct=60, window=1, glue={blocks[3][0]})   # 표 시작 앞 경계를 금지
        self.assertNotIn(blocks[3][0], [s for s, _ in spans])

    def test_caption_glue_from_doc_covers_both_caption_styles(self):
        doc = {"doc_id": "d", "dataset": "x", "fmt": "pdf", "lang": "en", "blocks": [
            {"id": "b0", "type": "paragraph", "text": "intro"},
            {"id": "b1", "type": "table", "text": "| a | b |", "caption": "Table 1. embedded caption"},
            {"id": "b2", "type": "table", "text": "| c | d |"},
            {"id": "b3", "type": "paragraph", "text": "Table 2. docling caption", "role": "caption", "for": "b2"},
            {"id": "b4", "type": "paragraph", "text": "Figure caption before", "role": "caption", "for": "b5"},
            {"id": "b5", "type": "table", "text": "| e | f |"},
            {"id": "b6", "type": "paragraph", "text": "unrelated"}]}
        text, pieces = render(doc)
        glue = caption_glue(doc, pieces)
        starts = {p.block_ids[0]: p.start for p in pieces if p.kind == "table"}
        by_text = {text[p.start:p.end][:10]: p.start for p in pieces}
        self.assertIn(starts["b1"], glue)                                 # 표의 caption 필드 -> 캡션 조각 + 표 조각
        self.assertIn(by_text["Table 2. d"], glue)                        # 표 -> 뒤따르는 Docling 캡션
        self.assertIn(starts["b5"], glue)                                 # 앞선 Docling 캡션 -> 표
        self.assertNotIn(by_text["unrelated"], glue)
        self.assertNotIn(starts["b2"], glue)


class Constraint5_ParagraphBoundaryPreferred(unittest.TestCase):
    def test_paragraph_boundary_gets_exactly_para_bonus_over_a_sentence_boundary(self):
        # 모든 조각이 같은 주제(벡터 동일)라 기본 거리가 같다 -> 문단 경계와 문장 경계의 차이는 보너스뿐이다
        long_para = " ".join(words("alpha", 15) for _ in range(3))        # 45단어 > max 40 -> 문장 조각 3개
        text, blocks = build([("para", long_para), ("para", words("alpha", 20))])
        ch = ConstrainedSmartChunker(Emb(), count_words, "pdf", "en", 10, 40, window=2, percentile=90, para_bonus=0.05)
        pieces = [p for sec in ch._sm.sections(text, blocks) for p in sec]
        self.assertEqual(len(pieces), 4)                                  # 문장 3개 + 문단 1개
        d = ch._distances(text, pieces, blocks)
        self.assertAlmostEqual(d[2] - d[0], 0.05, places=4)               # 문단 경계(s3|문단2) - 문장 경계(s1|s2)
        self.assertAlmostEqual(d[2] - d[1], 0.05, places=4)

    def test_cut_chosen_among_equal_meaning_ends_a_paragraph(self):
        long_para = " ".join(words("alpha", 15) for _ in range(3))
        parts = [("para", long_para), ("para", words("alpha", 20))]
        text, blocks, spans, _ = run(parts, mn=10, mx=40)
        self.assertIn(blocks[0][1], [e for _, e in spans])                # 문단 1의 끝에서 청크가 끝난다


class SemanticIsAlwaysUsed(unittest.TestCase):
    def test_section_below_max_is_still_split_at_a_topic_change(self):
        parts = [("heading", "# S")] + [("para", words("alpha", 10)) for _ in range(5)] + [("para", words("beta", 10)) for _ in range(5)]
        text, blocks, spans, ch = run(parts, mn=20, mx=200)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[1][0], blocks[6][0])                       # beta 문단부터 새 청크
        self.assertEqual(ch.stats["semantic_cuts"], 1)
        legacy = SmartChunker(Emb(), count_words, "pdf", "en", 20, 200).chunk_spans(text, blocks)
        self.assertEqual(len(legacy), 1)                                  # 대조: legacy는 섹션이 max 이하라 자르지 않는다


class SizeRegularization(unittest.TestCase):
    def test_oversized_group_is_cut_at_the_largest_distance_with_both_sides_above_min(self):
        parts = [("para", words("alpha", 20)) for _ in range(3)] + [("para", words("beta", 20)) for _ in range(3)]
        text, blocks, spans, ch = run(parts, mn=20, mx=70, pct=100)       # pct 100 -> 의미 컷 없이 크기 정리만 작동
        sizes = [count_words(text[s:e]) for s, e in spans]
        self.assertTrue(all(20 <= x <= 70 for x in sizes), sizes)
        self.assertIn(blocks[3][0], [s for s, _ in spans])                # 주제가 바뀌는 곳이 컷에 포함

    def test_unavoidable_violation_is_reported_not_hidden(self):
        # 마지막 섹션(제목 + 짧은 문단)은 min 미만이고 앞 청크와 합치면 max를 넘는다
        parts = [("heading", "# A"), ("para", words("alpha", 58)), ("heading", "# B"), ("para", words("beta", 8))]
        text, blocks, spans, ch = run(parts, mn=20, mx=60)
        self.assertEqual(len(spans), 2)
        self.assertEqual(ch.stats["under_min"], 1)


class SrtMode(unittest.TestCase):
    def test_sentence_continuing_into_next_cue_is_not_cut(self):
        cues = ["The cab driver took the wheel.", "He said the cab was fast.", "The driver smiled at me.",
                "Then the driver asked me if", "I wanted pizza with cheese.", "The oven was already hot.",
                "Cheese melted in the oven.", "Pizza smelled great."]
        text, blocks, pos = "", [], 0
        for c in cues:
            blocks.append((pos, pos + len(c), "cue"))
            text += c + "\n"
            pos += len(c) + 1

        class E2(Emb):
            def encode(self, texts, normalize_embeddings=True):
                out = []
                for t in texts:
                    low = t.lower()
                    v = np.array([len(re.findall("cab|driver|wheel", low)), len(re.findall("pizza|cheese|oven", low))], float) + 0.05
                    out.append(v / np.linalg.norm(v))
                return np.array(out)
        ch = ConstrainedSmartChunker(E2(), count_words, "srt", "en", 6, 22, window=1, percentile=80)
        spans = ch.chunk_spans(text, blocks)
        self.assertNotIn(blocks[3][1], [e for _, e in spans])             # 큐 4("...asked me if")에서 끝나는 청크가 없다
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], blocks[-1][1])


class IntegrationWithDocuments(unittest.TestCase):
    def test_invariants_on_fixture_documents(self):
        emb = Emb()
        models = {"embed_model": emb, "count": count_words}
        for name in ("example_a", "example_c"):
            doc = load_doc(HERE / "fixtures" / f"{name}.json")
            cfg = {"mode": "pdf", "algorithm": "constrained", "min_tokens": 20, "max_tokens": 60, "window": 2, "percentile": 90}
            chunks, info = splitters.chunk(doc, "smart", cfg, models)
            text = to_text(doc)
            self.assertTrue(chunks)
            for c in chunks:
                self.assertEqual(c["text"], text[c["start"]:c["end"]])    # 리터럴 슬라이스
            for a, b in zip(chunks, chunks[1:]):
                self.assertLessEqual(a["end"], b["start"])
            kept = re.sub(r"\s+", "", "".join(c["text"] for c in chunks))
            self.assertEqual(kept, re.sub(r"\s+", "", text))              # 내용 보존
            types = {b["id"]: b["type"] for b in doc["blocks"]}
            self.assertTrue(all(types[c["block_ids"][-1]] != "heading" for c in chunks))   # 제목으로 끝나는 청크 없음
            self.assertIn("smart_chunks", info)
            self.assertEqual(chunks, splitters.chunk(doc, "smart", cfg, models)[0])         # 결정적

    def test_runner_tag_and_default_algorithm(self):
        from chunking.runner import tag_for
        from common.config import dataset_cfg
        self.assertEqual(dataset_cfg("vectara")["chunkers"]["smart"]["algorithm"], "constrained")
        self.assertNotIn("algorithm", dataset_cfg("srt")["chunkers"]["smart"])      # srt는 legacy 유지(앞서 정한 대로)
        self.assertEqual(tag_for("smart", {"algorithm": "constrained", "percentile": 90, "min_tokens": 100, "max_tokens": 400, "window": 2}),
                         "smart__c-p90-min100-max400-w2")
        self.assertEqual(tag_for("smart", {"min_tokens": 100, "max_tokens": 400, "window": 2}), "smart__min100-max400-w2")


if __name__ == "__main__":
    unittest.main()
