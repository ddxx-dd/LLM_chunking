import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from config import SRT_KOR_DIR, SRT_ENG_DIR, ALLGANIZE_DIR, RESULTS_DIR, SUBTITLE_DATASETS
from srt.loader import load_srt
from docx_track.loader import load_docx
from splitters import split_sentences, calculate_similarities, calculate_threshold

_LOAD_BY_SUFFIX = {".srt": load_srt, ".docx": load_docx}


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

_noah_en, _noah_ko = SUBTITLE_DATASETS["Noah"]
# 예전엔 docx_eng의 Canada_Government 보고서를 썼는데 코퍼스가 allganize로 교체되며 삭제됨
# - 현재 코퍼스(allganize, finance 도메인)의 문서 하나로 교체.
_allganize_doc = "한-호주_퇴직연금_포럼_책자_최종_.docx"
FILES = [
    SRT_KOR_DIR / _noah_ko,
    SRT_ENG_DIR / _noah_en,
    ALLGANIZE_DIR / "docx" / "finance" / _allganize_doc,
]
# matplotlib 기본 폰트(DejaVu Sans)가 한글 글리프를 지원하지 않아 그래프 제목이
# 깨지는 걸 막기 위해, 그래프에 넣을 라벨만 영문으로 매핑한다(저장 파일명은 원래대로).
TITLE_LABELS = {
    _noah_ko: "Noah (KOR subtitles)",
    _noah_en: "Noah (ENG subtitles)",
    _allganize_doc: "Allganize Finance Doc (KR-AUS Pension Forum)",
}
METHOD = "percentile"
AMOUNT = 10


def main():
    print("모델 로드중..")
    model = SentenceTransformer("BAAI/bge-m3")
    print("로드 완료\n")

    for FILE_PATH in FILES:
        if not FILE_PATH.exists():
            continue
        doc = _LOAD_BY_SUFFIX[FILE_PATH.suffix.lower()](str(FILE_PATH))
        text = doc.page_content
        print("파일:", FILE_PATH.name, "(글자수:", len(text), ")")

        sentences = split_sentences(text, 200)
        sentence_texts = [s[0] for s in sentences]
        vectors = model.encode(sentence_texts, show_progress_bar=False)

        similarities = calculate_similarities(vectors)
        threshold = calculate_threshold(similarities, METHOD, AMOUNT)

        save_name = FILE_PATH.stem + "_" + METHOD + str(AMOUNT) + ".png"
        save_path = str(RESULTS_DIR / save_name)

        title_label = TITLE_LABELS.get(FILE_PATH.name, FILE_PATH.stem)
        plot_boundaries(similarities, threshold, save_path, title=f"{title_label} - {METHOD}")
        print("저장 완료:", save_path, "\n")


if __name__ == "__main__":
    main()
