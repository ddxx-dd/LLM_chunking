import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR, SRT_ENG_DIR, DOCX_KOR_DIR, RESULTS_DIR
from preprocessing.loader import load_file
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import split_sentences, calculate_similarities, calculate_threshold, split_at_boundaries


def plot_boundaries(similarities, threshold, save_path, title=""):
    """문장 간 유사도 곡선 + 임계값 + 절단 지점을 그려서 저장."""
    plt.figure(figsize=(14, 5))
    plt.plot(similarities, linewidth=0.8, color="steelblue", label="similarity")
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold = {threshold:.3f}")
    cut_x = [i for i, s in enumerate(similarities) if s < threshold]
    cut_y = [similarities[i] for i in cut_x]
    plt.scatter(cut_x, cut_y, color="red", s=25, zorder=3, label=f"cut points ({len(cut_x)})")
    plt.xlabel("sentence index")
    plt.ylabel("cosine similarity")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

FILES = [
    SRT_KOR_DIR / "트루먼쇼.srt",
    SRT_ENG_DIR / "The_Truman_Show_Eng.srt",
    DOCX_KOR_DIR / "3-1.큐.docx",
]
# matplotlib 기본 폰트(DejaVu Sans)가 한글 글리프를 지원하지 않아 그래프 제목이
# 깨지는 걸 막기 위해, 그래프에 넣을 라벨만 영문으로 매핑한다(저장 파일명은 원래대로).
TITLE_LABELS = {
    "트루먼쇼.srt": "Truman Show (KOR subtitles)",
    "The_Truman_Show_Eng.srt": "Truman Show (ENG subtitles)",
    "3-1.큐.docx": "Queue Lecture Notes (3-1)",
}
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

    title_label = TITLE_LABELS.get(FILE_PATH.name, FILE_PATH.stem)
    plot_boundaries(similarities, threshold, save_path, title=f"{title_label} - {METHOD}")
    print("저장 완료:", save_path, "\n")