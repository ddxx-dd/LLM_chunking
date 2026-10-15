"""preprocess.stage2 규칙: 전처리 검증에서 실제로 나온 오류 사례가 다시 생기지 않는지(모델·데이터 없이 돈다)."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import stage2  # noqa: E402


def blk(i, text, typ="paragraph", page=1, **kw):
    return {"id": f"b{i}", "type": typ, "text": text, "page": page, **kw}


class TestCaption(unittest.TestCase):
    def is_cap(self, t):
        return bool(stage2.CAP_SHAPE.match(stage2.AXIS_LEAD.sub("", t)))

    def test_caption_shapes(self):
        for t in ("Fig. 4 : Results", "Figure 2. Overview", "Figure A.1. Event study", "fig. S2. The band gap",
                  "Extended Data Fig. 1: Floquet", "10 0 10 1 FIG. 4. Gapless roton"):
            self.assertTrue(self.is_cap(t), t)

    def test_body_sentences_are_not_captions(self):
        # 옛 판별은 'Figure 4.5 illustrates …'의 '4.'을 캡션 번호로 보고 본문을 지웠다(4편 6블록)
        for t in ("Figure 4.5 illustrates the plot of the interface", "Figure 3 shows the results", "Fig. 5 detailed inspection"):
            self.assertFalse(self.is_cap(t), t)


class TestRunningHeader(unittest.TestCase):
    def page_block(self, i, text, page):
        return blk(i, text, "heading", page, boxes=[[page, 50, 30, 300, 40]], page_size=[600, 800])

    def test_alternating_identical_header_removed(self):
        bs = [self.page_block(i, "Consistent time travel for market making", p) for i, p in enumerate((4, 6, 8))]
        self.assertEqual(stage2.running_ids(bs, 20), {0, 1, 2})

    def test_numbered_headings_kept(self):
        # 숫자를 빼고 묶으면 22·24·26쪽의 'Proof of Lemma 2/4/7' 제목을 머리말로 지웠다
        bs = [self.page_block(i, f"Proof of Lemma {n}", p) for i, (n, p) in enumerate(((2, 22), (4, 24), (7, 26)))]
        self.assertEqual(stage2.running_ids(bs, 40), set())


class TestFirstPageNote(unittest.TestCase):
    def test_affiliation_and_funding(self):
        for t in ("Abbas Acar and Selcuk Uluagac are with the Cyber-Physical Systems Security Lab, Florida International University.",
                  "This work was partially supported by the National Science Foundation.",
                  "This paper was recommended for publication by Editor M. Ani Hsieh."):
            self.assertTrue(stage2.FIRST_PAGE_NOTE.search(t), t)

    def test_note_run_does_not_eat_later_pages(self):
        # Docling이 여러 쪽을 한 블록으로 합친 문서에서 교신 저자 줄 뒤의 초록~5쪽 본문까지 지웠다
        bs = [blk("0.0", "*Corresponding author."), blk("0.1", "Abstract This study introduces a package that lowers barriers. " * 5),
              blk("0.2", "1. Motivation and significance The growing body of literature ...", page=2)]
        kept = stage2.remove_blocks(bs, Counter(), {"n_pages": 5, "pictures": []})
        self.assertEqual([b["id"] for b in kept], ["b0.1", "b0.2"])


class TestReferences(unittest.TestCase):
    def test_body_footnote_after_refs_kept(self):
        # 문서 끝으로 옮긴 본문 각주가 참고문헌 영역 뒤에 놓여 함께 지워졌다
        bs = [blk(0, "References", "heading"), blk(1, "Meyer, A., Ante, L., 2020. Effects of coin offerings. J. Finance 2, 259-283."),
              blk(2, "13 We excluded draws from the models, consistent with literature (Edmans et al., 2007; Palomino et al., 2009).", "footnote")]
        kept = stage2.remove_blocks(bs, Counter(), {"n_pages": 5, "pictures": []})
        self.assertEqual([b["id"] for b in kept], ["b2"])

    def test_appendix_after_refs_kept(self):
        # 제목 없이 참고문헌 뒤에 붙은 부록(수식)이 참고문헌으로 지워졌다
        bs = [blk(0, "References", "heading"), blk(1, "[1] A. Smith, Some paper, Phys. Rev. Lett. 1, 2 (2020)."),
              blk(2, "u_t = f(u)", "formula"), blk(3, "On the one hand, for D = 1, this PDE is the one presented in [26].")]
        kept = stage2.remove_blocks(bs, Counter(), {"n_pages": 5, "pictures": []})
        self.assertEqual([b["id"] for b in kept], ["b2", "b3"])


class TestFormula(unittest.TestCase):
    def test_trim_neighbor_duplicate(self):
        # 수식 인식이 다음 문장까지 받아 적어 같은 문장이 두 번 나왔다
        f = "h ≲ ∑ ( ∫ | D f | ) with θ < 1 so if l is even then the x integral on the right hand side consists of integer order derivatives"
        nxt = "with θ ′ i + θ ′′ i < 1. So, if l is even, then the x -integral on the right hand side consists of integer order derivatives and"
        out = stage2.trim_neighbor_dup(f, "Proof. Let us first verify (121).", nxt, Counter())
        self.assertNotIn("integer order", out)
        self.assertTrue(out.startswith("h ≲ ∑"))

    def test_plain_formula_untouched(self):
        self.assertEqual(stage2.trim_neighbor_dup("f(x) = max { 0 , x } for all x", "We define the function", "where x is real.", Counter()),
                         "f(x) = max { 0 , x } for all x")


class TestTableCaption(unittest.TestCase):
    def test_caption_goes_to_following_table(self):
        # 가장 가까운 표를 고르면 '<표 2>'가 앞의 '<표 1>' 표에 붙었다(판결문: 뒤 표는 다음 쪽, 캡션 앞에 본문 문단).
        # 캡션이 앞 표 바로 뒤에 붙어 있으면 '표 아래 캡션' 양식과 구분되지 않아 앞 표에 붙는다(알려진 한계).
        doc = {"blocks": [blk(0, "<표 1> 일반현황", page=2), blk(1, "| a | b |\n| - | - |\n| 1 | 2 |", "table", 2),
                          blk(2, "원고의 일반현황은 아래 <표 2>와 같다.", page=2), blk(3, "<표 2> 원고 일반현황", page=2),
                          blk(4, "| c | d |\n| - | - |\n| 3 | 4 |", "table", 3)],
               "n_pages": 3, "pictures": []}
        units, _ = stage2.build_units(doc, lambda ts: [len(t.split()) for t in ts])
        tables = [u["text"] for u in units if u["kind"] == "table"]
        self.assertTrue(tables[0].startswith("<표 1>") and "<표 2>" not in tables[0])
        self.assertTrue(tables[1].startswith("<표 2>"))


if __name__ == "__main__":
    unittest.main()
