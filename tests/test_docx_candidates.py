"""docx_corpus 근거 구간 샘플러(qa_build/docx_candidates) 단위 테스트 - 공백 단위 토큰 카운터, 합성 문서."""
import unittest

from support import count_words
from common.docjson import render
from qa_build import docx_candidates as D

KW = dict(min_tokens=40, max_tokens=120)


def para(i, n=30, word="w"):
    letters = lambda x: "".join(chr(97 + int(d)) for d in str(x))             # 숫자 -> 알파벳 (글자 비율 필터를 통과하는 단어)
    text = " ".join(f"{word}{letters(i)}{letters(k)}" for k in range(n)) + "."
    return {"id": f"b{i}", "type": "paragraph", "text": text[0].upper() + text[1:], "heading_path": [], "page": None}   # 문장처럼 대문자로 시작


def heading(i, text, level=1):
    return {"id": f"b{i}", "type": "heading", "level": level, "text": text, "heading_path": [], "page": None}


def table(i, text):
    return {"id": f"b{i}", "type": "table", "text": text, "caption": "", "heading_path": [], "page": None}


def make_doc(blocks, doc_id="D"):
    return {"doc_id": doc_id, "dataset": "docx_corpus", "fmt": "docx", "lang": "en", "blocks": blocks}


def body(n=40):
    blocks = []
    for i in range(n):
        blocks.append(heading(len(blocks), f"Section {i}") if i % 8 == 0 else para(len(blocks)))
    return blocks


class Sampling(unittest.TestCase):
    def sample(self, doc, **kw):
        return D.sample_doc(doc, count_words, per_doc=kw.pop("per_doc", 5), **{**KW, **kw})

    def test_deterministic_and_seed_dependent(self):
        doc = make_doc(body())
        a, _ = self.sample(doc)
        b, _ = self.sample(doc)
        self.assertEqual(a, b)
        c, _ = self.sample(doc, seed=7)
        self.assertNotEqual([x["start"] for x in a], [x["start"] for x in c])

    def test_other_documents_do_not_change_a_documents_candidates(self):
        a, _ = self.sample(make_doc(body(), "A"))
        b, _ = self.sample(make_doc(body(), "A"))
        self.assertEqual(a, b)                                           # 시드는 (SEED, doc_id)뿐이다

    def test_spans_are_literal_slices_within_length_and_not_overlapping(self):
        doc = make_doc(body())
        text, _ = render(doc)
        cands, _ = self.sample(doc)
        self.assertEqual(len(cands), 5)
        for c in cands:
            self.assertEqual(c["text"], text[c["start"]:c["end"]])
            self.assertTrue(KW["min_tokens"] <= count_words(c["text"]) <= KW["max_tokens"], c["n_tokens"])
            self.assertEqual(c["n_tokens"], count_words(c["text"]))
        for a, b in zip(cands, cands[1:]):
            self.assertLessEqual(a["end"], b["start"])
        self.assertEqual([c["cid"] for c in cands], [f"D#{k}" for k in range(5)])

    def test_lengths_vary_across_the_range(self):
        cands, _ = self.sample(make_doc(body(120)), per_doc=15)
        sizes = [c["n_tokens"] for c in cands]
        self.assertGreater(max(sizes) - min(sizes), 30, sizes)

    def test_span_never_starts_or_ends_with_a_heading(self):
        doc = make_doc(body())
        text, _ = render(doc)
        for c in self.sample(doc)[0]:
            self.assertFalse(c["text"].startswith("#"))
            self.assertFalse(c["text"].rstrip().split("\n")[-1].startswith("#"))

    def test_long_paragraph_is_cut_at_sentence_boundaries(self):
        sentences = " ".join(f"Sentence {i} has a few words in it." for i in range(60))
        doc = make_doc([para(0, 5), {"id": "b1", "type": "paragraph", "text": sentences, "heading_path": [], "page": None}, para(2, 5)])
        text, _ = render(doc)
        cands, _ = self.sample(doc, per_doc=2)
        self.assertTrue(cands)
        for c in cands:
            self.assertLessEqual(c["n_tokens"], KW["max_tokens"])


class TableCandidates(unittest.TestCase):
    def doc_with_table(self, cells):
        blocks = body(30)
        blocks.insert(14, table(99, cells))
        return make_doc(blocks)

    def test_documents_with_tables_get_table_anchored_candidates(self):
        doc = self.doc_with_table("| name | value |\n| --- | --- |\n| alpha | 10 |\n| beta | 20 |\n| gamma | 30 |")
        cands, _ = D.sample_doc(doc, count_words, **KW)
        anchors = [c for c in cands if c["anchor"] == "table"]
        self.assertGreaterEqual(len(anchors), 1)
        self.assertLessEqual(len(anchors), D.TABLE_PER_DOC)
        for c in anchors:
            self.assertTrue(c["has_table"])
            self.assertGreater(c["table_ratio"], 0)

    def test_document_without_tables_has_no_table_candidates(self):
        cands, _ = D.sample_doc(make_doc(body()), count_words, **KW)
        self.assertTrue(all(c["anchor"] == "random" and not c["has_table"] and c["table_ratio"] == 0 for c in cands))

    def test_mostly_empty_table_is_not_used_as_anchor(self):
        doc = self.doc_with_table("|  |  |  |\n| --- | --- | --- |\n|  |  | x |\n|  |  |  |")
        cands, _ = D.sample_doc(doc, count_words, **KW)
        self.assertFalse(any(c["has_table"] for c in cands))

    def test_single_table_is_not_picked_twice(self):
        doc = self.doc_with_table("| name | value |\n| --- | --- |\n| alpha | 10 |")
        cands, _ = D.sample_doc(doc, count_words, **KW)
        self.assertEqual(len(cands), 5)
        self.assertEqual(sum(c["has_table"] for c in cands), 1)


class ExtraSamples(unittest.TestCase):
    """보강 표본: 이미 쓴 구간 피하기, 표 시작만, 큰 표 통째로."""
    def big_table_doc(self, rows):
        cells = "\n".join(f"| name{k} | {' '.join(['value'] * 4)} {k} |" for k in range(rows))
        blocks = body(30)
        blocks.insert(14, table(99, "| name | value |\n| --- | --- |\n" + cells))
        return make_doc(blocks)

    def test_avoid_keeps_new_candidates_off_existing_spans(self):
        doc = make_doc(body(60))
        first, _ = D.sample_doc(doc, count_words, per_doc=4, **KW)
        avoid = [(c["start"], c["end"]) for c in first]
        more, _ = D.sample_doc(doc, count_words, per_doc=4, avoid=avoid, cid_tag="a", **KW)
        self.assertTrue(more)
        for c in more:
            self.assertTrue(all(c["end"] <= a or c["start"] >= b for a, b in avoid))
            self.assertRegex(c["cid"], r"#a\d+$")

    def test_table_only_gives_only_table_anchored_candidates(self):
        doc = make_doc(body(30) + [table(98, "| name | value |\n| --- | --- |\n| alpha | 10 |\n| beta | 20 |\n| gamma | 30 |")])
        cands, _ = D.sample_doc(doc, count_words, table_only=True, cid_tag="t", **KW)
        self.assertTrue(all(c["anchor"] == "table" and c["has_table"] for c in cands))
        none, _ = D.sample_doc(make_doc(body(30)), count_words, table_only=True, **KW)
        self.assertEqual(none, [])

    def test_big_table_is_taken_whole_only_above_the_normal_limit(self):
        doc = self.big_table_doc(60)
        text, _ = render(doc)
        got = D.big_tables(doc, count_words, min_tokens=KW["max_tokens"] + 1, max_tokens=5000)
        self.assertEqual(len(got), 1)
        c = got[0]
        self.assertGreater(c["n_tokens"], KW["max_tokens"])
        self.assertEqual(text[c["start"]:c["end"]], c["text"])
        self.assertTrue(c["text"].startswith("| name | value |") and c["text"].rstrip().endswith("59 |"))
        self.assertEqual(D.big_tables(self.big_table_doc(2), count_words, min_tokens=KW["max_tokens"] + 1, max_tokens=5000), [])      # 작은 표는 대상 아님
        self.assertEqual(D.big_tables(doc, count_words, avoid=[(c["start"], c["end"])], min_tokens=KW["max_tokens"] + 1, max_tokens=5000), [])


class CutSpans(unittest.TestCase):
    """구간이 문장·목록 중간에서 끊기지 않게 하는 검증."""

    def doc(self, texts, head=True):
        blocks = [heading(0, "Rules")] if head else []
        blocks += [{"id": f"b{len(blocks) + k}", "type": "paragraph", "text": t, "heading_path": [], "page": None} for k, t in enumerate(texts)]
        return make_doc(blocks)

    def atoms(self, doc):
        text, pieces = render(doc)
        return text, D._atoms(text, pieces, count_words, 120)

    def test_start_and_end_predicates(self):
        text, at = self.atoms(self.doc(["A person must not act unless:", "(a) there is a safety case; and", "(b) it covers the activity.",
                                         "continues the sentence above.", "A full sentence ends here."]))
        k = {t: i for i, t in enumerate(text[a["start"]:a["end"]] for a in at)}
        self.assertTrue(D._end_open(at, k["A person must not act unless:"], text))               # 콜론으로 끝남
        self.assertTrue(D._end_open(at, k["(a) there is a safety case; and"], text))              # 접속어로 끝남
        self.assertTrue(D._start_open(at, k["(b) it covers the activity."], text))                # 앞 문장이 있는 번호 항목
        self.assertTrue(D._start_open(at, k["continues the sentence above."], text))              # 소문자로 시작
        self.assertTrue(D._end_open(at, k["(b) it covers the activity."], text))                  # 다음 조각이 소문자로 이어짐
        self.assertFalse(D._start_open(at, k["A full sentence ends here."], text))
        self.assertFalse(D._end_open(at, k["A full sentence ends here."], text))

    def test_number_directly_after_a_heading_is_a_clause_start_not_a_cut(self):
        text, at = self.atoms(self.doc(["(1) A person must register within 30 days."]))
        self.assertFalse(D._start_open(at, 1, text))

    def test_spans_are_extended_to_include_the_intro_and_the_rest_of_a_list(self):
        doc = self.doc(["A person must not act unless:", "(a) there is a safety case in force for the facility; and",
                        "(b) the safety case provides for the activity concerned.", "An unrelated closing sentence stays outside."])
        text, at = self.atoms(doc)
        i, j = D._complete(at, 2, 2, text, count_words, 120)                    # (a) 항목만 있던 구간
        self.assertEqual((i, j), (1, 3))
        self.assertTrue(text[at[i]["start"]:at[j]["end"]].startswith("A person must not act unless:"))
        self.assertTrue(text[at[i]["start"]:at[j]["end"]].endswith("concerned."))

    def test_extension_that_exceeds_the_limit_is_rejected(self):
        doc = self.doc(["A person must not act unless:", "(a) " + " ".join(["wordy"] * 100) + "; and", "(b) the end."])
        text, at = self.atoms(doc)
        self.assertIsNone(D._complete(at, 3, 3, text, count_words, 60))

    def test_sampled_spans_never_start_or_end_cut(self):
        texts = []
        for n in range(24):
            texts.append("The rule number %s says the following applies:" % "x" * (n % 3 + 1))
            texts.append("(a) " + " ".join(["alpha"] * 12) + "; and")
            texts.append("(b) " + " ".join(["beta"] * 12) + ".")
            texts.append("A plain closing sentence number %d with enough words to count." % n)
        doc = make_doc(self.doc(texts)["blocks"], "D")
        text, _ = render(doc)
        cands, _ = D.sample_doc(doc, count_words, per_doc=5, min_tokens=30, max_tokens=90)
        self.assertTrue(cands)
        for c in cands:
            first, last = c["text"].lstrip()[:1], c["text"].rstrip()
            self.assertFalse(first.islower() or first == "(", c["text"][:40])
            self.assertFalse(last.endswith((":", ";", ",", "and")), last[-30:])

    def test_malformed_tables_are_not_ok(self):
        self.assertTrue(D._table_ok("| a | b |\n| - | - |\n| 1 | 2 |"))
        self.assertFalse(D._table_ok("| a | b |\n| - | - |"))                               # 행 없음
        self.assertFalse(D._table_ok("| a | b |\n| - | - |\n| 1 | 2 | 3 |"))                  # 열 수 불일치


class Filters(unittest.TestCase):
    def test_spans_under_reference_headings_are_dropped(self):
        blocks = [{**b, "heading_path": ["References"]} for b in body(40)]
        cands, rejected = D.sample_doc(make_doc(blocks), count_words, **KW)
        self.assertEqual(cands, [])
        self.assertGreater(rejected.get("heading", 0), 0)

    def test_dot_leader_and_symbol_spans_are_dropped(self):
        self.assertEqual(D._passes_filters("Introduction to the program and its goals ..... 5\nBackground and methods of the study ..... 9", [], []), "dotleader")
        self.assertEqual(D._passes_filters("1 2 3 4 5 6 7 8 9 10 11 12 13 14", [], []), "alpha")
        self.assertIsNone(D._passes_filters("A normal sentence about funding rules.", [], ["Funding"]))

    def test_only_the_nearest_heading_decides_for_contents_but_references_apply_to_subsections(self):
        # Docling이 "Table of Contents"를 닫지 않아 본문 전체가 그 아래 경로가 된 문서: 가장 가까운 제목이 본문이면 통과
        self.assertIsNone(D._passes_filters("A normal sentence about funding rules.", [], ["Table of Contents", "Policies"]))
        self.assertEqual(D._passes_filters("A normal sentence about funding rules.", [], ["Policies", "Contents"]), "heading")
        self.assertEqual(D._passes_filters("A normal sentence about funding rules.", [], ["References", "Journal articles"]), "heading")

    def test_table_of_contents_spans_are_dropped(self):
        toc = "\n\n".join(f"Article {i} - SOME TITLE\t{i + 2}" for i in range(12))
        self.assertEqual(D._passes_filters(toc, [], []), "toc")
        titled = "Table of Contents\n\nThis report describes funding rules for the program in detail and its aims."
        self.assertEqual(D._passes_filters(titled, [], []), "toc")
        body = "(1)\tA person must not engage in the activity unless a safety case is in force for the facility concerned."
        self.assertIsNone(D._passes_filters(body, [], []))

    def test_locate_and_stale_report_after_preprocessing_changes(self):
        import json, shutil, tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
        try:
            doc = make_doc(body(30), "D")
            (d / "D.json").write_text(json.dumps(doc), encoding="utf-8")
            cands, _ = D.sample_doc(doc, count_words, **KW)
            cands = {c["cid"]: c for c in cands}
            self.assertEqual(D.stale_report(cands, d), (sorted(cands), {}, []))          # 그대로면 전부 ok
            # 전처리가 바뀌어 앞쪽 블록이 하나 사라졌다고 하자 -> 그 뒤 구간의 위치만 밀린다
            changed = make_doc(doc["blocks"][1:], "D")
            (d / "D.json").write_text(json.dumps(changed), encoding="utf-8")
            ok, moved, lost = D.stale_report(cands, d)
            self.assertEqual(len(ok) + len(moved) + len(lost), len(cands))
            self.assertTrue(moved)
            text = render(changed)[0]
            for cid, (s, e) in moved.items():
                self.assertEqual(text[s:e], cands[cid]["text"])                             # 새 위치가 맞다
            fixed = D.relocate(cands, moved, d)
            for cid in moved:
                self.assertEqual(text[fixed[cid]["start"]:fixed[cid]["end"]], fixed[cid]["text"])
                self.assertTrue(fixed[cid]["block_ids"])
            self.assertEqual(D.stale_report(fixed, d)[2], lost)                              # 잃은 것은 그대로 잃은 것
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_locate_rejects_missing_and_ambiguous_text(self):
        self.assertEqual(D.locate("aa bb cc", "bb"), (3, 5))
        self.assertIsNone(D.locate("aa bb bb", "bb"))
        self.assertIsNone(D.locate("aa bb", "zz"))
        self.assertEqual(D.locate("aa  bb\ncc", "aa bb cc"), (0, 9))                        # 공백 차이는 무시

    def test_candidates_json_round_trip(self):
        import json, shutil, tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
        try:
            cands, _ = D.sample_doc(make_doc(body()), count_words, **KW)
            path = D.write_candidates({c["cid"]: c for c in cands}, d / "meta" / "candidates.json")
            self.assertEqual(sorted(json.loads(path.read_text(encoding="utf-8"))), [c["cid"] for c in cands])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
