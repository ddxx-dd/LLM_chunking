import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import SRT_KOR_DIR, DOCX_KOR_DIR, RESULTS_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import (
    split_sentences, calculate_similarities,
    calculate_threshold, split_at_boundaries,
)
from analysis.analyzer import plot_boundaries

#설정
FILES = [SRT_KOR_DIR / "부산행.srt", DOCX_KOR_DIR / "스택.docx"]
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10


# ---- 모델 로드 ----
print("모델 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print("로드 완료\n")

for FILE_PATH in FILES:

    # 로드
    text = load_file(FILE_PATH)
    print("파일: " + FILE_PATH.name)
    print("원본: " + str(len(text)) + "글자\n")

    # 청킹
    fixed_chunks = fixed_chunking(text, CHUNK_SIZE, OVERLAP)

    sentences = split_sentences(text, 200)
    vectors = model.encode(sentences)
    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, METHOD, AMOUNT)
    semantic_chunks = split_at_boundaries(sentences, similarities, threshold)

    #시각화
    save_name = FILE_PATH.stem + "_" + METHOD + str(AMOUNT) + ".png"

    plot_boundaries(
        similarities,threshold,
        str(RESULTS_DIR / save_name),
        title = str(FILE_PATH.name) + " — " + METHOD + " (amount=" + str(AMOUNT) + ")"
    )


    print("-" * 70 + "\n")
    print(FILE_PATH.name + " — " + METHOD + " (amount=" + str(AMOUNT) + ") 파일 생성 완료")
    print("-" * 70 + "\n")


