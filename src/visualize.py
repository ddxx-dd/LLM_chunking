from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import SRT_KOR_DIR
from loader import load_file
from fixed_chunker import fixed_chunking
from semantic_chunker import (
    split_sentences, calculate_similarities,
    calculate_threshold, split_at_boundaries,
)
from analyzer import plot_boundaries

#설정
FILE_PATH = SRT_KOR_DIR / "부산행.srt"
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10
OUT_DIR = "../results/"

#로드
text = load_file(FILE_PATH)
print("파일: " + FILE_PATH.name)
print("원본: " + str(len(text)) + "글자\n")

print("모델 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print("로드 완료\n")


#청킹
fixed_chunks = fixed_chunking(text,CHUNK_SIZE,OVERLAP)

sentences = split_sentences(text,200)
vectors = model.encode(sentences)
similarities = calculate_similarities(vectors)
threshold = calculate_threshold(similarities,METHOD,AMOUNT)
semantic_chunks = split_at_boundaries(sentences,similarities,threshold)

print("단순 분할: " + str(len(fixed_chunks)) + "개")
print("의미 분할: " + str(len(semantic_chunks)) + "개\n")


#시각화
plot_boundaries(
    similarities,threshold,
    OUT_DIR + "baseline.png",
    title = "semantic_chunking (" + METHOD + " " + str(AMOUNT) + ")"
)


