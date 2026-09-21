import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EMBED_MODEL, TOKENIZER
from llm import load_llm


def setup_models():
    """모든 파이프라인이 공유하는 디바이스/임베딩/LLM 로딩 - 여기 하나로 모아서
    각 파이프라인 스크립트가 "무슨 실험을 하는지"만 남도록 한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"실행 디바이스: {device.upper()}")
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    tokenizer, llm_model, device = load_llm(TOKENIZER, device)
    print("모델 준비 완료")
    return device, embed_model, tokenizer, llm_model
