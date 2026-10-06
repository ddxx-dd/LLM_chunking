"""docjson 자체 일관성 테스트 (docs/restructure_plan.md 4.1절 불변식, 12절 1단계).
기존 코드의 출력과 비교하지 않는다. 실행: .venv311/bin/python -m unittest discover -s tests -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path


from support import HERE, FIX
from common.docjson import (MD_SEP, SRT_SEP, Piece, block_spans, load_doc, render,
                            save_doc, to_lc_document, to_text)

A_SPANS = [(0, 16, "heading"), (18, 241, "para"), (243, 388, "para"), (390, 401, "heading"),
           (403, 501, "para"), (503, 602, "table"), (604, 937, "para"), (939, 955, "heading"),
           (957, 1069, "para")]


def md_doc(blocks):
    return {"doc_id": "t", "fmt": "pdf", "blocks": blocks}


class Invariants(unittest.TestCase):
    """불변식 ①~④를 두 예시 문서와 합성 문서에서 모두 확인한다."""

    def docs(self):
        yield load_doc(FIX / "example_a.json")
        yield load_doc(FIX / "example_b.json")
        yield md_doc([
            {"id": "h", "type": "heading", "level": 2, "text": "Title"},
            {"id": "l1", "type": "list_item", "text": "- one"},
            {"id": "l2", "type": "list_item", "text": "- two"},
            {"id": "t", "type": "table", "text": "| a |\n|---|\n| 1 |", "caption": "Table 1: x"},
            {"id": "f", "type": "formula", "text": "<!-- formula-not-decoded -->"},
            {"id": "p", "type": "paragraph", "text": "end."},
        ])

    def test_1_slices_match_pieces(self):
        # ① 조각의 [start:end] 슬라이스가 해당 블록 텍스트와 같다 (여기서는 비어 있지 않고 범위 안에 있음을 확인)
        for doc in self.docs():
            text, pieces = render(doc)
            for p in pieces:
                self.assertTrue(0 <= p.start < p.end <= len(text))
                self.assertTrue(text[p.start:p.end].strip())

    def test_2_ordered_and_non_overlapping(self):
        # ② 조각은 순서대로이며 서로 겹치지 않는다
        for doc in self.docs():
            _, pieces = render(doc)
            for a, b in zip(pieces, pieces[1:]):
                self.assertLessEqual(a.end, b.start)

    def test_3_block_ids_one_to_one(self):
        # ③ 텍스트가 있는 모든 블록 id가 조각에 나타난다 (표에 caption이 있으면 그 조각은 표 id를 공유)
        for doc in self.docs():
            _, pieces = render(doc)
            seen = [i for p in pieces for i in p.block_ids]
            expected = [b["id"] for b in doc["blocks"] if (b.get("text") or "").strip()]
            self.assertEqual(sorted(set(seen)), sorted(expected))
            for p in pieces:
                self.assertEqual(len(p.block_ids), 1)       # 조각 하나는 블록 하나(목록 항목도 따로)

    def test_4_deterministic(self):
        # ④ 같은 문서를 두 번 렌더해도 결과가 같다
        for doc in self.docs():
            self.assertEqual(render(doc), render(doc))


class ExampleA(unittest.TestCase):
    """계획서 5.3절 예시 A (논문 한 부분)."""

    def test_text_and_spans(self):
        doc = load_doc(FIX / "example_a.json")
        expected = (FIX / "example_a.txt").read_text(encoding="utf-8")
        self.assertEqual(to_text(doc), expected)
        self.assertEqual(len(expected), 1069)
        self.assertEqual(block_spans(doc), A_SPANS)

    def test_piece_slices_equal_block_text(self):
        doc = load_doc(FIX / "example_a.json")
        text, pieces = render(doc)
        by_id = {b["id"]: b for b in doc["blocks"]}
        for p in pieces:
            b = by_id[p.block_ids[0]]
            want = ("#" * b["level"] + " " + b["text"]) if b["type"] == "heading" else b["text"]
            self.assertEqual(text[p.start:p.end], want)

    def test_blocks_separated_by_blank_line(self):
        doc = load_doc(FIX / "example_a.json")
        text, pieces = render(doc)
        for a, b in zip(pieces, pieces[1:]):
            self.assertEqual(text[a.end:b.start], MD_SEP)
        self.assertEqual(pieces[-1].end, len(text))      # 끝에는 구분자를 붙이지 않는다


class ExampleB(unittest.TestCase):
    """계획서 5.3절 예시 B (Deadpool 1-19번 큐)."""

    def setUp(self):
        self.doc = load_doc(FIX / "example_b.json")

    def test_cue_offsets(self):
        text, pieces = render(self.doc)
        self.assertEqual(len(text), 691)
        self.assertEqual(len(pieces), 19)
        self.assertTrue(text.endswith(SRT_SEP))          # 마지막 큐 뒤에도 SEP
        pos = 0
        for p, b in zip(pieces, self.doc["blocks"]):
            self.assertEqual((p.start, p.end, p.kind), (pos, pos + len(b["text"]), "cue"))
            self.assertEqual(text[p.start:p.end], b["text"])
            pos += len(b["text"]) + len(SRT_SEP)
        self.assertEqual(pieces[0].end, 25)              # "Kinda lonesome back here."

    def test_lc_document_units_match_old_format(self):
        d = to_lc_document(self.doc)
        self.assertEqual(d.metadata["fmt"], "srt")
        self.assertEqual(d.metadata["name"], "Deadpool.en.srt")
        self.assertEqual(d.page_content, to_text(self.doc))
        units = d.metadata["units"]
        self.assertEqual(len(units), 19)
        u = units[0]
        self.assertEqual((u["start"], u["end"], u["kind"]), (0, 25, "cue"))
        self.assertEqual(u["meta"]["index"], 1)
        self.assertAlmostEqual(u["meta"]["t_start"], 180.430, places=3)
        self.assertAlmostEqual(u["meta"]["t_end"], 182.850, places=3)
        for unit, b in zip(units, self.doc["blocks"]):
            self.assertEqual(d.page_content[unit["start"]:unit["end"]], b["text"])

    def test_lc_document_only_for_srt(self):
        with self.assertRaises(ValueError):
            to_lc_document(load_doc(FIX / "example_a.json"))


class RenderRules(unittest.TestCase):
    def test_each_list_item_is_its_own_piece(self):
        # 목록 항목은 항목마다 별도 조각(빈 줄로 구분): 참고문헌 수백 개가 쪼개지지 않는 한 조각이 되지 않게
        doc = md_doc([{"id": "a", "type": "paragraph", "text": "intro"},
                      {"id": "l1", "type": "list_item", "text": "[1] first ref."},
                      {"id": "l2", "type": "list_item", "text": "[2] second ref."},
                      {"id": "b", "type": "paragraph", "text": "outro"}])
        text, pieces = render(doc)
        self.assertEqual(text, "intro\n\n[1] first ref.\n\n[2] second ref.\n\noutro")
        self.assertEqual([(p.kind, p.block_ids) for p in pieces],
                         [("para", ("a",)), ("para", ("l1",)), ("para", ("l2",)), ("para", ("b",))])

    def test_footnote_renders_as_para_in_block_order(self):
        # 각주는 전처리가 블록 목록 맨 끝으로 옮겨 둔다. render는 블록 순서대로 para로 내보낸다.
        doc = md_doc([{"id": "a", "type": "paragraph", "text": "body."},
                      {"id": "fn", "type": "footnote", "text": "* Corresponding author", "after_block_id": "a"}])
        text, pieces = render(doc)
        self.assertEqual(text, "body.\n\n* Corresponding author")
        self.assertEqual([(p.kind, p.block_ids) for p in pieces], [("para", ("a",)), ("para", ("fn",))])

    def test_table_caption_is_separate_para_piece_before_table(self):
        doc = md_doc([{"id": "t", "type": "table", "text": "| a |\n|---|\n| 1 |", "caption": "Table 1: x"}])
        text, pieces = render(doc)
        self.assertEqual(text, "Table 1: x\n\n| a |\n|---|\n| 1 |")
        self.assertEqual([p.kind for p in pieces], ["para", "table"])
        self.assertEqual(text[pieces[0].start:pieces[0].end], "Table 1: x")

    def test_table_without_caption(self):
        doc = md_doc([{"id": "t", "type": "table", "text": "| a |"}])
        self.assertEqual([p.kind for p in render(doc)[1]], ["table"])

    def test_empty_blocks_skipped(self):
        doc = md_doc([{"id": "a", "type": "paragraph", "text": "   "},
                      {"id": "b", "type": "heading", "level": 1, "text": ""},
                      {"id": "c", "type": "paragraph", "text": "ok"}])
        text, pieces = render(doc)
        self.assertEqual(text, "ok")
        self.assertEqual(pieces, [Piece(0, 2, "para", ("c",))])

    def test_empty_doc(self):
        self.assertEqual(render(md_doc([])), ("", []))

    def test_formula_and_code_are_para(self):
        doc = md_doc([{"id": "f", "type": "formula", "text": "E=mc^2"},
                      {"id": "c", "type": "code", "text": "print(1)"}])
        self.assertEqual([p.kind for p in render(doc)[1]], ["para", "para"])

    def test_unknown_block_type_raises(self):
        with self.assertRaises(ValueError):
            render(md_doc([{"id": "x", "type": "figure", "text": "?"}]))

    def test_srt_with_non_cue_raises(self):
        with self.assertRaises(ValueError):
            render({"doc_id": "s", "fmt": "srt", "blocks": [{"id": "p", "type": "paragraph", "text": "x"}]})

    def test_headings_use_level(self):
        doc = md_doc([{"id": "h", "type": "heading", "level": 3, "text": "T"}])
        self.assertEqual(to_text(doc), "### T")


class SaveLoad(unittest.TestCase):
    def test_round_trip_keeps_korean(self):
        doc = {"doc_id": "k", "fmt": "pdf", "title": "한국어 제목",
               "blocks": [{"id": "b0", "type": "paragraph", "text": "한글 본문입니다."}]}
        tmp = tempfile.mkdtemp(dir=HERE)                  # 프로젝트 밖(/tmp)을 쓰지 않는다
        try:
            path = Path(tmp) / "sub" / "k.json"
            save_doc(doc, path)
            self.assertEqual(load_doc(path), doc)
            self.assertIn("한글 본문입니다.", path.read_text(encoding="utf-8"))   # ensure_ascii=False
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
