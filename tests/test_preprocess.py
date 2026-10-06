"""전처리 단위 테스트 - Docling 항목은 가짜 객체로 대신한다(실제 변환은 scripts/preprocess.py 파일럿으로 확인).
실행: .venv311/bin/python -m unittest discover -s tests -v
"""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


from support import HERE
from docling_core.types.doc import DocItemLabel as L, GroupLabel

from common.docjson import render
from preprocess._docling import build_blocks
from preprocess.docx_prep import make_heading_level
from preprocess import pdf_prep
from preprocess.pdf_prep import PdfHeadingLevel
from preprocess.srt_prep import normalize, parse_srt_text, to_sec
from preprocess.validate import validate_doc


class FakeItem:
    def __init__(self, label, text="", page=1, ref=None, parent=None, level=None, captions=None, md=""):
        self.label, self.text, self.self_ref, self.parent, self.level = label, text, ref, parent, level
        self.prov = [SimpleNamespace(page_no=page)] if page else []
        self.captions = captions or []
        self._md = md

    def export_to_markdown(self, doc=None):
        return self._md


class FakeDoc:
    def __init__(self, items):
        self.items = items

    def iterate_items(self):
        return ((it, 0) for it in self.items)


def inline_parent(n):
    return SimpleNamespace(cref=f"#/groups/{n}", resolve=lambda doc: SimpleNamespace(label=GroupLabel.INLINE))


def blocks_of(items, heading_level=lambda it: 1):
    return build_blocks(FakeDoc(items), heading_level, table_to_md=lambda it: it.export_to_markdown())


def hl_by_level(it):
    return it.level or 1


class HeadingLevel(unittest.TestCase):
    def test_numbered_and_named(self):
        h = PdfHeadingLevel()
        f = lambda t: h(SimpleNamespace(text=t))
        self.assertEqual(f("Abstract"), 1)           # 번호 없음 + 앞에 번호 제목 없음 -> 1
        self.assertEqual(f("1 Introduction"), 1)
        self.assertEqual(f("2.1 Setup"), 2)
        self.assertEqual(f("2.1.3 Detail"), 3)
        self.assertEqual(f("References"), 1)          # 최상위 절 이름
        self.assertEqual(f("Remark"), 3)              # 번호 없는 제목: 직전 번호 제목(3단계)의 아래, 최대 3
        self.assertEqual(f("3. Method"), 1)           # 끝의 점 무시

    def test_unnumbered_after_level_one_goes_to_two(self):
        h = PdfHeadingLevel()
        h(SimpleNamespace(text="4 Results"))
        self.assertEqual(h(SimpleNamespace(text="Ablation")), 2)

    def test_docx_levels_shift_when_title_present(self):
        title = FakeItem(L.TITLE, "T", level=None)
        s1 = FakeItem(L.SECTION_HEADER, "A", level=1)
        s2 = FakeItem(L.SECTION_HEADER, "B", level=2)
        with_title = make_heading_level(FakeDoc([title, s1, s2]))
        self.assertEqual([with_title(title), with_title(s1), with_title(s2)], [1, 2, 3])
        without = make_heading_level(FakeDoc([s1, s2]))
        self.assertEqual([without(s1), without(s2)], [1, 2])
        deep = FakeItem(L.SECTION_HEADER, "D", level=9)
        self.assertEqual(without(deep), 6)            # 최대 6


class BuildBlocks(unittest.TestCase):
    def test_types_paths_and_skips(self):
        items = [
            FakeItem(L.SECTION_HEADER, "1 Intro", level=1),
            FakeItem(L.TEXT, "Body one."),
            FakeItem(L.PICTURE, ""), FakeItem(L.PAGE_HEADER, "run head"), FakeItem(L.PAGE_FOOTER, "3"),
            FakeItem(L.SECTION_HEADER, "1.1 Detail", level=2),
            FakeItem(L.LIST_ITEM, "- a"), FakeItem(L.LIST_ITEM, "- b"),
            FakeItem(L.FORMULA, ""), FakeItem(L.TEXT, "   "),
            FakeItem(L.SECTION_HEADER, "2 Next", level=1),
            FakeItem(L.TEXT, "After."),
        ]
        blocks, log = blocks_of(items, hl_by_level)
        self.assertEqual([b["type"] for b in blocks],
                         ["heading", "paragraph", "heading", "list_item", "list_item", "heading", "paragraph"])
        self.assertEqual([b["id"] for b in blocks], [f"b{i}" for i in range(7)])
        self.assertEqual(blocks[3]["heading_path"], ["1 Intro", "1.1 Detail"])
        self.assertEqual(blocks[6]["heading_path"], ["2 Next"])            # 같은 수준 제목이 앞 제목을 대체
        self.assertEqual(log["skipped"], {"PICTURE": 1, "PAGE_HEADER": 1, "PAGE_FOOTER": 1})
        self.assertEqual(blocks[0]["page"], 1)

    def test_footnote_goes_to_end_and_ids_are_remapped(self):
        items = [FakeItem(L.TEXT, "First."), FakeItem(L.FOOTNOTE, "* note one", page=1),
                 FakeItem(L.TEXT, "Second."), FakeItem(L.FOOTNOTE, "2 note two", page=2),
                 FakeItem(L.TEXT, "Third.")]
        blocks, log = blocks_of(items)
        self.assertEqual([b["type"] for b in blocks], ["paragraph"] * 3 + ["footnote"] * 2)
        self.assertEqual([b["text"] for b in blocks][3:], ["* note one", "2 note two"])
        self.assertEqual([b["id"] for b in blocks], ["b0", "b1", "b2", "b3", "b4"])
        self.assertEqual(blocks[3]["after_block_id"], "b0")                # 원래 위치(First. 뒤)
        self.assertEqual(blocks[4]["after_block_id"], "b1")
        self.assertEqual(log["footnotes"], 2)
        text, pieces = render({"fmt": "pdf", "blocks": blocks})            # 평문에서도 맨 끝
        self.assertTrue(text.endswith("* note one\n\n2 note two"))

    def test_footnote_before_any_block(self):
        blocks, _ = blocks_of([FakeItem(L.FOOTNOTE, "lead"), FakeItem(L.TEXT, "Body.")])
        self.assertEqual([b["type"] for b in blocks], ["paragraph", "footnote"])
        self.assertIsNone(blocks[1]["after_block_id"])

    def test_caption_after_table_is_linked(self):
        items = [FakeItem(L.TEXT, "Intro."),
                 FakeItem(L.TABLE, md="| a |\n|---|\n| 1 |", captions=[SimpleNamespace(cref="#/texts/9")]),
                 FakeItem(L.CAPTION, "Table 1: x.", ref="#/texts/9"),
                 FakeItem(L.CAPTION, "Figure 1: y.", ref="#/texts/10")]
        blocks, log = blocks_of(items)
        table, cap, fig = blocks[1], blocks[2], blocks[3]
        self.assertEqual((table["type"], cap["type"], cap["role"], cap["for"]), ("table", "paragraph", "caption", table["id"]))
        self.assertEqual((fig["role"], fig.get("for")), ("caption", None))
        self.assertNotIn("caption", table)                                 # 표에는 caption 필드를 쓰지 않는다
        self.assertEqual(log["docling_tables"], 1)

    def test_caption_before_table_is_linked(self):
        items = [FakeItem(L.CAPTION, "Table 2: z.", ref="#/texts/5"),
                 FakeItem(L.TABLE, md="| b |", captions=[SimpleNamespace(cref="#/texts/5")])]
        blocks, _ = blocks_of(items)
        self.assertEqual(blocks[0]["for"], blocks[1]["id"])

    def test_inline_group_merged_into_one_paragraph(self):
        p = inline_parent(1)
        items = [FakeItem(L.TEXT, "Part one", parent=p), FakeItem(L.TEXT, "part two.", parent=p),
                 FakeItem(L.TEXT, "Next paragraph.", parent=inline_parent(2))]
        blocks, log = blocks_of(items)
        self.assertEqual([b["text"] for b in blocks], ["Part one part two.", "Next paragraph."])
        self.assertEqual(log["inline_groups_merged"], 1)

    def test_whitespace_collapsed_and_empty_table_counted(self):
        items = [FakeItem(L.TEXT, "A      B"), FakeItem(L.TABLE, md="   "), FakeItem(L.TABLE, md="| x |      | y |")]
        blocks, log = blocks_of(items)
        self.assertEqual([b["text"] for b in blocks], ["A B", "| x | | y |"])
        self.assertEqual((log["docling_tables"], log["empty_tables"]), (2, 1))

    def test_empty_heading_dropped(self):
        blocks, log = blocks_of([FakeItem(L.SECTION_HEADER, "  ", level=1), FakeItem(L.TEXT, "x")])
        self.assertEqual([b["type"] for b in blocks], ["paragraph"])
        self.assertEqual(log["empty_headings"], 1)


def good_pdf_doc():
    blocks = [{"id": "b0", "type": "heading", "level": 1, "text": "1 Intro", "page": 1},
              {"id": "b1", "type": "paragraph", "text": "x" * 3000, "page": 1},
              {"id": "b2", "type": "table", "text": "| a |", "page": 2},
              {"id": "b3", "type": "footnote", "text": "* n", "page": 1}]
    return {"fmt": "pdf", "n_pages": 2, "blocks": blocks, "log": {"docling_tables": 1, "empty_tables": 0}}


class Validate(unittest.TestCase):
    def test_good_doc(self):
        v = validate_doc(good_pdf_doc())
        self.assertEqual(v["errors"], [])
        self.assertEqual((v["stats"]["n_tables"], v["stats"]["n_footnotes"]), (1, 1))

    def test_errors(self):
        d = good_pdf_doc(); d["blocks"][1]["id"] = "b0"
        self.assertTrue(any("중복" in e for e in validate_doc(d)["errors"]))
        d = good_pdf_doc(); d["blocks"][3], d["blocks"][2] = d["blocks"][2], d["blocks"][3]      # 각주가 중간에
        self.assertTrue(any("각주" in e for e in validate_doc(d)["errors"]))
        d = good_pdf_doc(); d["log"]["docling_tables"] = 2
        self.assertTrue(any("표 개수" in e for e in validate_doc(d)["errors"]))
        d = good_pdf_doc(); d["blocks"][2]["page"] = 9
        self.assertTrue(any("page" in e for e in validate_doc(d)["errors"]))
        d = good_pdf_doc(); d["blocks"][1]["text"] = " "
        self.assertTrue(any("빈 블록" in e for e in validate_doc(d)["errors"]))
        self.assertTrue(any("블록이 없다" in e for e in validate_doc({"fmt": "pdf", "blocks": []})["errors"]))

    def test_caption_duplicated_inside_table_is_error(self):
        d = good_pdf_doc()
        d["blocks"][2]["text"] = "Table 1: Conditions for the coexistence equilibria.\n\n| a |"
        d["blocks"].insert(3, {"id": "bc", "type": "paragraph", "role": "caption", "for": "b2", "page": 2,
                               "text": "Table 1: Conditions for the coexistence equilibria."})
        self.assertTrue(any("캡션이 중복" in e for e in validate_doc(d)["errors"]))

    def test_warnings(self):
        d = good_pdf_doc(); d["blocks"][1]["text"] = "short"
        self.assertTrue(any("글자 수가 적다" in w for w in validate_doc(d)["warnings"]))
        d = good_pdf_doc(); d["blocks"] = [b for b in d["blocks"] if b["type"] != "heading"]
        d["log"] = {}
        self.assertTrue(any("제목 블록이 없다" in w for w in validate_doc(d)["warnings"]))
        d = good_pdf_doc(); d["n_pages"] = 20                      # 글이 있는 쪽이 거의 없음
        self.assertTrue(any("글이 거의 없는 쪽" in w for w in validate_doc(d)["warnings"]))


SRT = ("﻿1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>Hello</i>  there\r\n\r\n"
       "2\r\n00:00:03.000 --> 00:00:04.000\r\n<i></i>\r\n\r\n"
       "3\r\n00:00:05,000 --> 00:00:06,000 X1:1\r\nline one\r\nline\xa0two\r\n")


class Srt(unittest.TestCase):
    def test_parse(self):
        blocks, log = parse_srt_text(SRT.lstrip("﻿"))
        self.assertEqual(log, {"cues_found": 3, "dropped_empty": 1})
        self.assertEqual([b["text"] for b in blocks], ["Hello there", "line one line two"])
        self.assertEqual([b["id"] for b in blocks], ["c1", "c2"])           # 블록 id는 순번
        self.assertEqual([b["index"] for b in blocks], [1, 3])               # index는 원본 번호
        self.assertAlmostEqual(blocks[0]["end"], 2.5)
        self.assertAlmostEqual(blocks[1]["start"], 5.0)

    def test_helpers(self):
        self.assertAlmostEqual(to_sec("01:02:03.250"), 3723.25)
        self.assertEqual(normalize("  a​  b\t c  "), "a b c")

    def test_validate_srt_counts(self):
        blocks, log = parse_srt_text(SRT.lstrip("﻿"))
        doc = {"fmt": "srt", "blocks": blocks, "log": log}
        self.assertEqual(validate_doc(doc)["errors"], [])
        doc["log"] = {"cues_found": 5, "dropped_empty": 0}
        self.assertTrue(any("큐 개수" in e for e in validate_doc(doc)["errors"]))




def make_pdf(path, text="Hello clean test"):
    import pymupdf
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), text)
    d.save(str(path))
    d.close()


class PdfClean(unittest.TestCase):
    """PDF 정제(PyMuPDF clean) 로직 - docs/restructure_plan.md 4.2절. 임시 파일은 tests/ 아래에 만들고 지운다."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=HERE))
        self.src = self.tmp / "a.pdf"
        make_pdf(self.src)
        self.work = self.tmp / "work"
        self.work.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_prepare_applies_clean_and_keeps_text(self):
        import pymupdf
        path, info = pdf_prep.prepare_pdf(self.src, self.work, enabled=True)
        self.assertEqual(path, self.work / "a.clean.pdf")
        self.assertTrue(path.exists())
        self.assertTrue(info["applied"] and info["enabled"] and "error" not in info)
        self.assertTrue(info["tool"].startswith("pymupdf-"))
        self.assertGreater(info["clean_bytes"], 0)
        doc = pymupdf.open(str(path))
        try:
            self.assertEqual(doc.page_count, 1)
            self.assertIn("Hello clean test", doc[0].get_text())
        finally:
            doc.close()

    def test_prepare_disabled_returns_original(self):
        path, info = pdf_prep.prepare_pdf(self.src, self.work, enabled=False)
        self.assertEqual(path, self.src)
        self.assertEqual((info["enabled"], info["applied"]), (False, False))
        self.assertEqual(list(self.work.iterdir()), [])

    def test_prepare_falls_back_to_original_on_failure(self):
        bad = self.tmp / "bad.pdf"
        bad.write_bytes(b"this is not a pdf")
        path, info = pdf_prep.prepare_pdf(bad, self.work, enabled=True)
        self.assertEqual(path, bad)
        self.assertFalse(info["applied"])
        self.assertIn("error", info)

    def test_page_count_mismatch_is_an_error(self):
        # clean_pdf가 쪽 수가 달라진 정제본을 만들면 예외 -> prepare_pdf가 원본으로 되돌린다
        with mock.patch.object(pdf_prep, "clean_pdf", side_effect=ValueError("정제본의 쪽 수가 다르다")):
            path, info = pdf_prep.prepare_pdf(self.src, self.work, enabled=True)
        self.assertEqual(path, self.src)
        self.assertIn("쪽 수", info["error"])

    # ---- convert_pdf: 가짜 변환기로 "어떤 경로가 변환기에 전달되는가"를 확인
    def _convert(self, **kw):
        seen = {}

        class FakeConverter:
            def convert(self_inner, path):
                seen["path"] = Path(path)
                seen["existed"] = Path(path).exists()
                return SimpleNamespace(document=SimpleNamespace(pages={1: object()}))

        entry = {"doc_id": "a", "dataset": "t", "lang": "en", "path": self.src, "source_file": "data/raw/a.pdf"}
        block = {"id": "b0", "type": "paragraph", "text": "x"}
        with mock.patch.object(pdf_prep, "build_blocks", return_value=([block], {})), \
                mock.patch.object(pdf_prep, "drop_corner_items", return_value=0):
            doc = pdf_prep.convert_pdf(entry, FakeConverter(), workdir=self.work, **kw)
        return doc, seen

    def test_convert_passes_cleaned_file_and_removes_it(self):
        doc, seen = self._convert()
        self.assertEqual(seen["path"].name, "a.clean.pdf")
        self.assertTrue(seen["existed"])                                   # 변환 중에는 정제본이 있었고
        self.assertFalse(seen["path"].exists())                            # 변환 뒤에는 지워졌다
        self.assertEqual(list(self.work.iterdir()), [])                    # 작업 폴더에 찌꺼기가 없다
        self.assertEqual(doc["source_file"], "data/raw/a.pdf")             # 출처는 항상 원본 경로
        self.assertTrue(doc["log"]["pdf_clean"]["applied"])
        self.assertEqual(doc["parser_options"]["pdf_clean"], pdf_prep.CLEAN_OPTIONS)
        self.assertEqual(doc["n_pages"], 1)

    def test_convert_without_clean_uses_original(self):
        doc, seen = self._convert(clean=False)
        self.assertEqual(seen["path"], self.src)
        self.assertFalse(doc["log"]["pdf_clean"]["applied"])
        self.assertIsNone(doc["parser_options"]["pdf_clean"])

    def test_convert_falls_back_when_clean_fails(self):
        with mock.patch.object(pdf_prep, "clean_pdf", side_effect=RuntimeError("boom")):
            doc, seen = self._convert()
        self.assertEqual(seen["path"], self.src)                           # 정제 실패해도 변환은 원본으로 계속된다
        self.assertIn("boom", doc["log"]["pdf_clean"]["error"])
        self.assertIsNone(doc["parser_options"]["pdf_clean"])


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PreviousStats(unittest.TestCase):
    def test_prev_stats(self):
        mod = load_script("preprocess")
        tmp = Path(tempfile.mkdtemp(dir=HERE))
        try:
            p = tmp / "d.json"
            self.assertIsNone(mod.prev_stats(p))                            # 없으면 None
            p.write_text(json.dumps({"blocks": [{"text": "abc"}, {"text": "de"}, {"text": ""}]}), encoding="utf-8")
            self.assertEqual(mod.prev_stats(p), {"total_chars": 5, "n_blocks": 3})
            p.write_text("{not json", encoding="utf-8")
            self.assertIsNone(mod.prev_stats(p))                            # 깨진 파일은 None
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
