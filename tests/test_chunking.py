"""chunking: 문장 분리와 청커 4종(모델 없이 가짜 임베딩으로 돈다)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import advanced_semantic, fixed, recursive, semantic  # noqa: E402
from chunking.sentences import split_structured, split_text, split_units  # noqa: E402

NS = lambda t: "".join(t.split())
TOPICS = {"cat": 0, "dog": 0, "pet": 0, "stock": 1, "market": 1, "price": 1, "rain": 2, "cloud": 2}


def fake_encode(texts):
    """주제 낱말 수를 센 벡터 + 문장마다 고정된 작은 잡음(길이 1). 주제가 같으면 코사인이 높고, 같은 주제 안에서도 조금씩 다르다."""
    out = []
    for t in texts:
        v = np.full(8, 0.05)
        for w in t.lower().replace(".", " ").split():
            if w in TOPICS:
                v[TOPICS[w]] += 1
        rng = np.random.default_rng(sum(map(ord, t)))
        v[4:] += rng.uniform(0, 0.25, 4)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


def count_words(texts):
    return [len(t.split()) for t in texts]


PETS = "The cat is a pet. A dog is a pet too. My cat and dog play. Every pet needs care."
MARKET = "The stock market fell today. Stock price moves daily. The market price rose again."


class TestSentences(unittest.TestCase):
    def test_paragraph_heading_table_formula(self):
        text = "I. INTRODUCTION\nGames model agents. They act.\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\n$$ x = 1 $$\n\nWe stop here."
        self.assertEqual(split_units(text), [("sentence", "I. INTRODUCTION"), ("sentence", "Games model agents."), ("sentence", "They act."),
                                             ("table", "| a | b |\n| - | - |\n| 1 | 2 |"), ("formula", "$$ x = 1 $$"), ("sentence", "We stop here.")])

    def test_blank_line_is_boundary(self):
        # 라이브러리만 쓰면 빈 줄을 넘어 '… plans This work was supported …'가 한 문장이 됐다
        self.assertEqual(split_text("each robot typically plans\n\nThis work was supported by ONR."),
                         ["each robot typically plans", "This work was supported by ONR."])

    def test_title_lines_kept_apart(self):
        # HotpotQA 원문: 'Passage 1:', 문서 제목이 본문 첫 문장에 붙었다
        self.assertEqual(split_text("Passage 1:\nTrusty system (prison)\nThe trusty system was a penitentiary system."),
                         ["Passage 1:", "Trusty system (prison)", "The trusty system was a penitentiary system."])

    def test_wrapped_lines_joined(self):
        # NarrativeQA 원문: 한 줄 약 70자로 접힌 줄은 이어서 문장을 나눈다
        out = split_text("Has he any objection to make to them which would justify him in\noverturning them? Was he not brought up\nby them?")
        self.assertEqual(out, ["Has he any objection to make to them which would justify him in overturning them?", "Was he not brought up by them?"])

    def test_structured_matches_split_text(self):
        text = "Title line\nBody one. Body two.\n\n| t | u |\n| - | - |\n\nLast para."
        items = split_structured(text)
        self.assertEqual([u for u, _, _ in items], split_text(text))
        self.assertEqual([s for _, s, _ in items], ["", "\n", " ", "\n\n", "\n\n"])
        self.assertEqual([h for _, _, h in items], [True, False, False, False, False])

    def test_text_preserved(self):
        text = "Title line\nBody one. Body two.\n\n| t | u |\n| - | - |\n\nLast para."
        self.assertEqual(NS("".join(split_text(text))), NS(text))


class TestFixedRecursive(unittest.TestCase):
    def test_fixed_characters(self):
        text = PETS + "\n\n" + MARKET
        out = fixed.chunk(text, chunk_size=40)
        self.assertTrue(all(len(c) <= 40 for c in out))
        self.assertEqual(NS("".join(out)), NS(text))

    def test_recursive_default(self):
        text = PETS + "\n\n" + MARKET
        out = recursive.chunk(text, chunk_size=90)
        self.assertTrue(all(len(c) <= 90 for c in out))
        self.assertEqual(out[0], PETS[:out[0].__len__()])


class TestSemantic(unittest.TestCase):
    def test_splits_at_topic_change(self):
        out = semantic.chunk(PETS + " " + MARKET, fake_encode, breakpoint_threshold_amount=80)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].endswith("Every pet needs care.") and out[1].startswith("The stock market"))


class TestAdvancedSemantic(unittest.TestCase):
    def run_(self, text, theta=100, **kw):
        kw.setdefault("beta", 0.3)
        return advanced_semantic.chunk(text, fake_encode, count_words, theta, return_ranges=True, **kw)

    def test_cuts_only_at_topic_change(self):
        chunks, units, ranges = self.run_(PETS + " " + MARKET)
        self.assertEqual(ranges, [(0, 4), (4, 7)])
        self.assertEqual(chunks[0], PETS)

    def test_beta_is_price_of_a_cut(self):
        # β가 아주 작으면 한 단위씩, 아주 크면 한 덩어리
        _, units, small = self.run_(PETS + " " + MARKET, beta=1e-4)
        _, _, large = self.run_(PETS + " " + MARKET, beta=50)
        self.assertEqual(len(small), len(units))
        self.assertEqual(large, [(0, len(units))])

    def test_theta_caps_chunk(self):
        _, units, ranges = self.run_(PETS, theta=10, beta=50)
        toks = count_words(units)
        self.assertTrue(all(sum(toks[i:j]) <= 10 or j - i == 1 for i, j in ranges))
        self.assertEqual([i for i, _ in ranges][1:], [j for _, j in ranges][:-1])          # 빈틈·겹침 없이 이어짐
        self.assertEqual(ranges[-1][1], len(units))

    def test_unit_bigger_than_theta_alone(self):
        _, units, ranges = self.run_("A cat and a dog and a pet sat together today. A cat.", theta=3)
        self.assertEqual(ranges[0], (0, 1))

    def test_min_tokens(self):
        # 주제가 바뀌어도 앞 청크가 min_tokens보다 작으면 자르지 않는다
        _, _, ranges = self.run_(PETS + " " + MARKET, min_tokens=25)
        self.assertGreater(ranges[0][1], 4)

    def test_no_cut_after_heading_and_format_kept(self):
        text = "Pets\nThe cat is a pet. A dog is a pet too.\n\nMarket\nThe stock market fell today. Stock price moves daily."
        chunks, units, ranges = self.run_(text, beta=1e-4)
        self.assertNotIn(1, [i for i, _ in ranges])                  # 'Pets' 바로 뒤에서 자르지 않음
        self.assertTrue(chunks[0].startswith("Pets\nThe cat"))
        whole, _, _ = self.run_(text, beta=50, theta=1000)
        self.assertEqual(whole, [text])                              # 이어 붙이면 원문 형식 그대로

    def test_paragraph_bonus(self):
        # 같은 주제 두 문단: 보너스가 크면 문단 경계에서 자른다
        text = "The cat is a pet. A dog is a pet too.\n\nMy cat and dog play. Every pet needs care."
        _, _, r0 = self.run_(text, beta=0.5, bonus_frac=0.0, theta=1000)
        _, _, r1 = self.run_(text, beta=0.5, bonus_frac=0.99, theta=1000)
        self.assertEqual(r0, [(0, 4)])
        self.assertEqual(r1, [(0, 2), (2, 4)])

    def test_target_size_search(self):
        text = " ".join([PETS, MARKET] * 4)
        chunks, units, ranges = advanced_semantic.chunk(text, fake_encode, count_words, 100, target=20, return_ranges=True)
        avg = sum(count_words(units)) / len(ranges)
        self.assertLess(abs(avg - 20), 8)


if __name__ == "__main__":
    unittest.main()
