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
    """참고문헌은 지우지 않고 ref로 표시한다(v4 형식). 표시 범위는 옛 삭제 범위와 같다."""

    def refs(self, bs):
        kept = stage2.remove_blocks(bs, Counter(), {"n_pages": 5, "pictures": []})
        return [b["id"] for b in kept], [b["id"] for b in kept if b.get("ref")]

    def test_body_footnote_after_refs_not_marked(self):
        # 문서 끝으로 옮긴 본문 각주가 참고문헌 영역 뒤에 놓여 함께 지워지던 실측 → 각주는 참고문헌이 아님
        bs = [blk(0, "References", "heading"), blk(1, "Meyer, A., Ante, L., 2020. Effects of coin offerings. J. Finance 2, 259-283."),
              blk(2, "13 We excluded draws from the models, consistent with literature (Edmans et al., 2007; Palomino et al., 2009).", "footnote")]
        kept, refs = self.refs(bs)
        self.assertEqual(kept, ["b0", "b1", "b2"])
        self.assertEqual(refs, ["b0", "b1"])

    def test_appendix_after_refs_not_marked(self):
        # 제목 없이 참고문헌 뒤에 붙은 부록(수식)이 참고문헌으로 지워지던 실측
        bs = [blk(0, "References", "heading"), blk(1, "[1] A. Smith, Some paper, Phys. Rev. Lett. 1, 2 (2020)."),
              blk(2, "u_t = f(u)", "formula"), blk(3, "On the one hand, for D = 1, this PDE is the one presented in [26].")]
        kept, refs = self.refs(bs)
        self.assertEqual(refs, ["b0", "b1"])

    def test_figure_caption_kept(self):
        kept, _ = self.refs([blk(0, "Figure 3: Results on the test set."), blk(1, "We now discuss the results in detail.")])
        self.assertEqual(kept, ["b0", "b1"])


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


class TestFormulaLatex(unittest.TestCase):
    def test_latex_wrapped(self):
        self.assertEqual(stage2.formula_text({"text": "x =  \\frac { a } { b }", "text_layer": "x = a/b"}, Counter()), "$$ x = \\frac { a } { b } $$")

    def test_trim_keeps_delimiters_and_tidies(self):
        # Q6이 수식 안에 반복된 다음 문장을 자르고 남은 '\\ \text {' 조각을 지운다(실측 2410.16441v2 Definition 1)
        f = "$$ J = \\frac { 1 } { 2 } x , ( 1 ) \\\\ \\text {and players follow a joint linear dynamical system} $$"
        out = stage2.trim_formula(f, "seeks to optimize a cost function:", "and players follow a joint linear dynamical system: x = A x", Counter())
        self.assertTrue(out.startswith("$$ ") and out.endswith(" $$"))
        self.assertNotIn("players", out)
        self.assertNotIn("\\text {", out)


class TestCleanText(unittest.TestCase):
    """preprocess.clean_text: v4 형식."""

    def setUp(self):
        from preprocess import clean_text
        self.ct = clean_text

    def u(self, text, kind="para", heading="", ids=("b0",), ref=False, formulas=()):
        return {"text": text, "kind": kind, "heading": heading, "block_ids": list(ids), "ref": ref, "formulas": [list(x) for x in formulas]}

    def test_references_one_paragraph_with_markers(self):
        units = [self.u("Intro text ends here."),
                 self.u("REFERENCES\nT. Basar, Game theory. SIAM, 1998.", heading="REFERENCES", ids=("b1", "b2"), ref=True),
                 self.u("D. Bertsekas, Dynamic programming. 2012.", "list", ids=("b3",), ref=True)]
        mk = {"T. Basar, Game theory. SIAM, 1998.": "[1]", "D. Bertsekas, Dynamic programming. 2012.": "[2]"}
        text, paras, _, ok = self.ct.build(units, {}, mk)
        self.assertEqual(text, "Intro text ends here.\n\nREFERENCES\n[1] T. Basar, Game theory. SIAM, 1998.\n[2] D. Bertsekas, Dynamic programming. 2012.")
        self.assertEqual(paras[1]["block_ids"], ["b1", "b2", "b3"])
        self.assertTrue(all(ok.values()))

    def test_formula_own_paragraph(self):
        t = "the cost function: $$ J = x $$ and players follow it."
        a = t.find("$$")
        text, paras, _, _ = self.ct.build([self.u(t, formulas=[(a, t.rfind("$$") + 2)])], {}, {})
        self.assertEqual(text, "the cost function:\n\n$$ J = x $$\n\nand players follow it.")
        self.assertEqual([m["kind"] for m in paras], ["para", "formula", "para"])

    def test_reference_table_unfolded_but_normal_table_kept(self):
        ref_tab = "References\n| [ACR19] | G. Azzi and L. Ribeiro. On the essence of conflicts. J. Log. Algebr. Methods, 2019. |\n| - | - |\n" \
                  "| [BHK22] | N. Behr and J. Krivine. Fundamentals of compositional rewriting theory, 2022. |\n| [McL92] | C. McLarty. Elementary categories, elementary toposes, 1992. |"
        normal = "Table B1. Known AGNs.\n| Fields | Sample |\n| - | - |\n| UDS | X-ray sources classified as AGNs in Chen et al. (2018) and others |\n" \
                 "| GOODS-S | all sources in Lyu et al. (2022) that meet the AGN criteria in one band |\n| GOODS-N | X-ray sources classified as AGNs in Xue et al. (2016) |"
        units = [self.u("Body text here.", ids=("b0",)), self.u(normal, "table", ids=("b1",)), self.u(ref_tab, "table", ids=("b2",))]
        text, paras, _, _ = self.ct.build(units, {}, {})
        self.assertEqual([m["kind"] for m in paras], ["para", "table", "references"])
        self.assertIn("[ACR19] G. Azzi", text)
        self.assertIn("| GOODS-S |", text)


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


class TestSentences(unittest.TestCase):
    """preprocess.sentences: 문장 나누기 함수(영어 nltk, 한국어 Kiwi + 구조 표시)."""

    def setUp(self):
        from preprocess import sentences
        self.s = sentences

    def test_english_abbreviations(self):
        out = self.s.split("Smith et al. (2020) showed this in Fig. 3 and Table 2. The value was 3.5 percent. It works.", "en")
        self.assertEqual(len(out), 3)

    def test_english_label_and_abbreviation(self):
        # 라벨만 남은 문장('Proof.', 'Lemma 2.1.')과 약어 뒤 소문자 시작('i.i.d. samples')이 따로 떨어졌다
        out = self.s.split("Lemma 2.1. Let X be finite. Proof. We use induction. Let U be i.i.d. samples from P.", "en")
        self.assertEqual(out, ["Lemma 2.1. Let X be finite.", "Proof. We use induction.", "Let U be i.i.d. samples from P."])

    def test_korean_structure_marks(self):
        out = self.s.split("□ 기술개발 지원 강화 * 도전적 문제 해결 과제 신설 ㅇ 인력 운영 효율화 추진", "ko")
        self.assertEqual(out, ["□ 기술개발 지원 강화", "* 도전적 문제 해결 과제 신설", "ㅇ 인력 운영 효율화 추진"])

    def test_korean_inline_footnote_mark_kept(self):
        # 문장 안 각주 표시('감축인력 * 의')는 자르지 않는다(판정 실측)
        self.assertEqual(len(self.s.split("비효율 분야 감축인력 * 의 국정과제 재배치", "ko")), 1)

class TestLongBench(unittest.TestCase):
    """preprocess.longbench: 원본을 그대로 읽는다(데이터 없이 도는 부분만)."""

    def test_script_detection(self):
        from preprocess import longbench
        self.assertTrue(longbench.is_script('<b>VENKMAN</b>\n<b>INT. ROOM</b>\n<P ID="dia">Hi</P><P>x</P>'))
        self.assertFalse(longbench.is_script("Produced by David Brannan.\n\nThe Adventure of the Devil's Foot"))


class TestParagraphsV5(unittest.TestCase):
    """preprocess.paragraphs(v5 문단 정리): 규칙마다 작은 예."""

    @staticmethod
    def run_(items, fmt="pdf", upto="ref_items"):
        from preprocess import paragraphs as P
        out, _ = P.tidy([{"text": t, "kind": k, "pages": [1], "block_ids": [f"b{i}"]} for i, (t, k) in enumerate(items)], fmt, upto)
        return [(p["text"], p["kind"]) for p in out], out

    def test_formula_joins_sentence(self):
        got, out = self.run_([("The flux is given by", "para"), ("$$ v = 1 $$", "formula"), ("where v is the speed.", "para"),
                              ("Next we show results.", "para")])
        self.assertEqual(got, [("The flux is given by $$ v = 1 $$ where v is the speed.", "para"), ("Next we show results.", "para")])
        self.assertEqual(out[0]["block_ids"], ["b0", "b1", "b2"])

    def test_list_with_lead(self):
        got, _ = self.run_([("The following may be specified:", "para"), ("recycled metal", "list"), ("low VOC finish", "list"), ("Refer to TR 01.", "para")])
        self.assertEqual(got[0], ("The following may be specified:\nrecycled metal\nlow VOC finish", "list"))

    def test_toc_lines(self):
        got, _ = self.run_([("A.1. General. 3-47", "para"), ("A.2. Scope. 3-47", "para"), ("A.3. Terms. 3-48", "para"), ("Body text here.", "para")])
        self.assertEqual(got[0], ("A.1. General. 3-47\nA.2. Scope. 3-47\nA.3. Terms. 3-48", "toc"))

    def test_heading_attached(self):
        got, _ = self.run_([("4. Discussion", "para"), ("The main purpose of this study was to establish a platform.", "para")])
        self.assertEqual(got, [("4. Discussion\nThe main purpose of this study was to establish a platform.", "para")])

    def test_broken_skips_caption(self):
        got, _ = self.run_([("They are computed by running the files", "para"), ("Fig. 1. A plot.", "caption"), ("(which contain the functions) and more.", "para")])
        self.assertEqual(got, [("They are computed by running the files (which contain the functions) and more.", "para"), ("Fig. 1. A plot.", "caption")])

    def test_interleaved_refs_move_to_end_pdf_only(self):
        body = [("Intro body sentence number %d that is long enough to count as body text here." % i, "para") for i in range(4)]
        refs = [("T. D. Lee and C.-N. Yang, Parity, Phys. Rev. 104, 254 (1956).", "list"), ("C. S. Wu, E. Ambler, Test, Phys. Rev. 105, 1413 (1957).", "list"),
                ("M. Kobayashi and T. Maskawa, CP violation, Phys. Rev. D 49, 652 (1973).", "list")]
        items = body + [("The estimator is limited because the noise in each", "para")] + refs + [("redshift bin could be subtracted.", "para"), ("Conclusion - We built it.", "para")]
        got, _ = self.run_(items, "pdf", upto="broken")             # v5 단계까지(v6 ref_items는 끝의 참고문헌을 항목마다 나눔)
        self.assertEqual(got[-1][1], "references")
        self.assertIn("each redshift bin could be subtracted.", got[-3][0])
        got_docx, _ = self.run_(items, "docx", upto="broken")
        self.assertNotEqual(got_docx[-1][1], "references")



class TestV6(unittest.TestCase):
    """clean_text v6(2026-10-26): 참고문헌 항목별 문단, 목차 보호 조건, 오류 수식 정리."""

    def test_ref_items_split_and_fragment(self):
        from preprocess import paragraphs as P
        t = ("References\nAcemoglu, D., Ozdaglar, A., and ParandehGheibi, A. (2010). Spread of information in social networks. "
             "Games and Economic Behavior ,\n70(2):194-227.\nAmico, M. and Keilegom, I. V. (2018). Cure models in survival analysis. Annual Review.")
        out = P.rule_ref_items([{"text": t, "kind": "references", "pages": [9], "block_ids": ["b1"]}], "pdf")
        self.assertEqual([p["text"] for p in out], [
            "References\nAcemoglu, D., Ozdaglar, A., and ParandehGheibi, A. (2010). Spread of information in social networks. Games and Economic Behavior , 70(2):194-227.",
            "Amico, M. and Keilegom, I. V. (2018). Cure models in survival analysis. Annual Review."])
        self.assertTrue(all(p["kind"] == "references" for p in out))

    def test_ref_items_protects_body(self):
        from preprocess import paragraphs as P
        t = "Preparation of alkaline extractant for the samples\nDissolve the following in 1L distilled water please:\n12 g sodium hydroxide"
        out = P.rule_ref_items([{"text": t, "kind": "references", "pages": [1], "block_ids": ["b1"]}], "docx")
        self.assertEqual([p["text"] for p in out], [t])                # 연도 있는 줄이 50% 미만 → 나누지 않음

    def test_toc_lines_protection(self):
        ok = stage2.toc_lines_are_toc
        self.assertTrue(ok(["1. Short title 2", "2. Commencement 2", "3. Terms used 3"], "1. Short title ... 2. Commencement ... 3. Terms used ...", False))
        self.assertFalse(ok(["Equality Act 2010", "Children Act 2014", "Data Protection Act 2018"], "", True))     # 연도로 끝남
        self.assertFalse(ok(["PMCY 6430E Biopharmaceutics 2", "PHAR 6200E Clinical Trials 4", "PHAR 7100E Biostatistics 3"],
                            "PMCY 6430E Biopharmaceutics PHAR 6200E Clinical Trials PHAR 7100E Biostatistics", True))   # 번호 감소(학점)
        self.assertFalse(ok(["‘male’ = 41", "‘female’ = 11", "Back to - Activity 2"], "", True))                  # '='
        self.assertFalse(ok(["Alpha section title 3", "Beta section title 4", "Gamma section title 5"], "unrelated body", False))  # 증거 없음

    def test_t2_only_docx(self):
        blocks = [{"type": "paragraph", "text": t} for t in ("Contents", "Intro part 1", "Methods part 2", "Results part 3")]
        self.assertEqual(stage2.toc_delete_ids(blocks, "pdf"), set())
        self.assertEqual(stage2.toc_delete_ids(blocks, "docx"), {1, 2, 3})

    def test_drop_toc_paras_keeps_body(self):
        text = "Body.\n\n가. 재직기간이 5년 미만인 사람의 퇴직급여: 4분의 1\n나. 재직기간이 5년 이상인 사람의 퇴직급여: 2분의 1\n다. 퇴직수당: 2분의 1\n\nMore body."
        paras = [{"kind": "para"}, {"kind": "toc"}, {"kind": "para"}]
        c, dropped = Counter(), []
        t2, p2 = stage2.drop_toc_paras(text, paras, c, dropped)
        self.assertEqual(t2, text)                                     # 증거 없는 목차 모양 본문(판결문)은 남김
        self.assertEqual(c["TC 목차 문단 보호(남김)"], 1)

    def test_formula_cut_to_layer_and_intertext(self):
        count = lambda ts: [len(t.split()) for t in ts]              # 낱말 수로 토큰을 대신(테스트용)
        cut = "$$ " + " + ".join(f"x _ {{ {i} }}" for i in range(80)) + " + \\sqrt { $$"     # 반복 없는 긴 수식이 '\\sqrt {'에서 끊김
        f, why = stage2.fix_formula(cut, {"text_layer": "x1 + x2 + ... + sqrt(y)"}, count, Counter())
        self.assertEqual((f, why), ("x1 + x2 + ... + sqrt(y)", "T"))
        ok = "$$ x = \\frac { a } { b } $$"
        self.assertEqual(stage2.fix_formula(ok, {"text_layer": "x = a/b"}, count, Counter()), (ok, None))
        s = stage2.strip_spaced_intertext("$$ x = 1 . \\\\ \\intertext { s u m a l f o r t h e a g e n t w i t h } $$")
        self.assertEqual(s, "$$ x = 1 . \\\\ $$")


if __name__ == "__main__":
    unittest.main()
