"""common.models: 인코딩 래퍼와 토큰 세기 (가짜 모델·토크나이저, GPU·모델 파일 없이)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import config  # noqa: E402
from common.models import load_embedder, make_encode, make_token_counter  # noqa: E402


class FakeModel:
    """fp16처럼 길이가 1에서 살짝 벗어난 벡터를 돌려준다."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append((len(texts), batch_size, normalize_embeddings))
        out = np.zeros((len(texts), config.EMBED_DIM), np.float16)
        for i, t in enumerate(texts):
            out[i, len(t) % config.EMBED_DIM] = 1.0002
            out[i, (len(t) + 1) % config.EMBED_DIM] = 0.01
        return out


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, texts, add_special_tokens):
        assert add_special_tokens is False                      # 특수 토큰은 세지 않는다
        return {"input_ids": [t.split() for t in texts]}


class TestEncode(unittest.TestCase):
    def test_float32_unit_norm_and_batch(self):
        m = FakeModel()
        v = make_encode(m, batch_size=7)(["a", "bb", "ccc"])
        self.assertEqual(v.dtype, np.float32)
        self.assertEqual(v.shape, (3, config.EMBED_DIM))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)   # 다시 정규화되어 정확히 1
        self.assertEqual(m.calls, [(3, 7, True)])

    def test_empty_input(self):
        m = FakeModel()
        v = make_encode(m)([])
        self.assertEqual(v.shape, (0, config.EMBED_DIM))
        self.assertEqual(m.calls, [])                           # 빈 입력이면 모델을 부르지 않는다

    def test_bad_precision(self):
        with self.assertRaises(ValueError):
            load_embedder("int8")


class TestTokenCounter(unittest.TestCase):
    def test_counts_without_special_tokens(self):
        tok = FakeTokenizer()
        count = make_token_counter(tok)
        self.assertEqual(count(["a b c", "", "one"]), [3, 0, 1])
        self.assertEqual(tok.model_max_length, 10 ** 9)          # 긴 글 경고를 끈다

    def test_many_texts_in_chunks(self):
        count = make_token_counter(FakeTokenizer())
        self.assertEqual(count(["x y"] * 5000), [2] * 5000)       # 4096개씩 나눠 세도 순서와 개수가 맞다


class TestConfig(unittest.TestCase):
    def test_paths_under_root(self):
        for p in (config.RAW_DIR, config.META2_DIR, config.CLEAN_DIR, config.QA_DIR, config.CHUNKS_DIR, config.INDEX_DIR, config.RESULTS_DIR):
            self.assertTrue(str(p).startswith(str(config.ROOT)))
        self.assertTrue((config.ROOT / "AGENTS.md").exists())


if __name__ == "__main__":
    unittest.main()
