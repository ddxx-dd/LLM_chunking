"""chunking: 문장 분리와 청커 4종(모델 없이 가짜 임베딩으로 돈다)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import advanced_semantic, fixed, recursive, semantic  # noqa: E402
from chunking.sentences import split_text, split_units  # noqa: E402

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
    def run_(self, text, theta, **kw):
        return advanced_semantic.chunk(text, fake_encode, count_words, theta, return_ranges=True, **kw)

    def test_first_change_inside_group(self):
        chunks, units, ranges = self.run_(PETS + " " + MARKET, theta=100)
        self.assertEqual(chunks[0], PETS)
        self.assertEqual(ranges[0], (0, 4))

    def test_theta_caps_chunk(self):
        chunks, units, ranges = self.run_(PETS, theta=10)
        toks = count_words(units)
        self.assertTrue(all(sum(toks[i:j]) <= 10 or j - i == 1 for i, j in ranges))
        self.assertEqual([i for i, _ in ranges][1:], [j for _, j in ranges][:-1])          # 빈틈·겹침 없이 이어짐
        self.assertEqual(ranges[-1][1], len(units))

    def test_unit_bigger_than_theta_alone(self):
        _, units, ranges = self.run_("A cat and a dog and a pet sat together today. A cat.", theta=3)
        self.assertEqual(ranges[0], (0, 1))

    def test_centroid_variants_cut_at_topic_change(self):
        # 주제가 바뀌는 곳(4번째 문장 뒤)에서는 반드시 자른다. 같은 주제 안에서도 응집도가 아주 높으면(σ가 작으면)
        # 유사도가 조금 낮은 문장('Every pet needs care.')에서 한 번 더 자를 수 있다(판단식의 성질, 파일럿에서 확인)
        for name in ("advanced_semantic1", "advanced_semantic2"):
            chunks, units, ranges = self.run_(PETS + " " + MARKET, theta=100, criterion=name)
            self.assertIn(4, [i for i, _ in ranges], name)
            self.assertEqual(ranges[-1][1], len(units), name)

    def test_centroid_leave_one_out(self):
        # 기존 단위는 자기를 뺀 중심과 비교한다: 같은 주제 단위는 이어지고 다른 주제는 끊긴다
        emb = fake_encode(["cat pet", "dog pet", "cat dog", "stock market"])
        j = advanced_semantic.Centroid(emb, [2, 2, 2, 2], k=0.0)
        j.reset(0); j.add(1)
        self.assertTrue(j.fits(2))
        j.add(2)
        self.assertFalse(j.fits(3))

    def test_para_units(self):
        chunks, units, _ = self.run_(PETS + "\n\n" + MARKET, theta=100, unit="para")
        self.assertEqual(units, [PETS, MARKET])
        self.assertEqual(chunks, [PETS, MARKET])


if __name__ == "__main__":
    unittest.main()
