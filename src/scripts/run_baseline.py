import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from analysis.analyzer import print_chunks, chunk_stats, print_stats

FILE_PATH = SRT_KOR_DIR / "부산행.srt"
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10

if FILE_PATH.exists():
    print("파일:", FILE_PATH.name)
    doc = load_file(FILE_PATH)
    text = doc.text
    print("텍스트 길이:", len(text), "자\n")

    print("모델 로드중..")
    model = SentenceTransformer("BAAI/bge-m3")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print("로드 완료\n")

    # 청킹 실행
    fixed = fixed_chunking(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    semantic = semantic_chunking(text, model, method=METHOD, amount=AMOUNT)

    # 청크 나열 출력
    print_chunks(doc, fixed, label="단순 분할 — 전체 청크", limit=5)
    print_chunks(doc, semantic, label="의미 기반 분할 — 전체 청크", limit=5)

    # 통계 출력
    fixed_stats = chunk_stats(doc, fixed, tokenizer=tokenizer)
    semantic_stats = chunk_stats(doc, semantic, tokenizer=tokenizer)

    print_stats(fixed_stats, label="단순 분할 요약")
    print_stats(semantic_stats, label="의미 기반 분할 요약")
else:
    print("파일을 찾을 수 없습니다:", FILE_PATH)