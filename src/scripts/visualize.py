import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR, DOCX_KOR_DIR, RESULTS_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import split_sentences, calculate_similarities, calculate_threshold, split_at_boundaries
from analysis.analyzer import plot_boundaries

FILES = [SRT_KOR_DIR / "부산행.srt", DOCX_KOR_DIR / "2-1.스택.docx"]
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10

print("모델 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
print("로드 완료\n")

for FILE_PATH in FILES:
    if not FILE_PATH.exists():
        continue
    doc = load_file(FILE_PATH)
    text = doc.text
    print("파일:", FILE_PATH.name, "(글자수:", len(text), ")")

    fixed_chunks = fixed_chunking(text, CHUNK_SIZE, OVERLAP)

    sentences = split_sentences(text, 200)
    sentence_texts = [s[0] for s in sentences]
    vectors = model.encode(sentence_texts, show_progress_bar=False)

    similarities = calculate_similarities(vectors)
    threshold = calculate_threshold(similarities, METHOD, AMOUNT)
    semantic_chunks = split_at_boundaries(sentences, similarities, threshold, text)

    save_name = FILE_PATH.stem + "_" + METHOD + str(AMOUNT) + ".png"
    save_path = str(RESULTS_DIR / save_name)

    plot_boundaries(similarities, threshold, save_path, title=FILE_PATH.name + " — " + METHOD)
    print("저장 완료:", save_path, "\n")