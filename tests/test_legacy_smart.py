"""legacy smart 테스트 (기본 경로에서는 쓰지 않는 보관 코드). 가짜 임베딩(키워드 축)과 공백 단위 토큰 카운터.
  SmartParts: 예전 SmartChunker(srt 설정이 아직 사용) / 나머지: 문단 단위 ConstrainedSmartChunker(이전 구현, 5개 제약)."""
import re
import unittest

import numpy as np


from support import FIX, TOPIC_AXES, make_stub, count_words, words, models_for
from chunking.legacy_smart import ConstrainedSmartChunker, SmartChunker, cue_continues, split_sentences
from chunking.units import caption_glue
from common.docjson import block_spans, load_doc, render, to_text



class Emb:
    """주제어(alpha/beta/gamma) 개수로 만든 벡터 - 같은 주제 글끼리 비슷하다."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in TOPIC_AXES], float) + 0.05
            out.append(v / np.linalg.norm(v))
        return np.array(out)


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


class LegacySmartParts(unittest.TestCase):
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
