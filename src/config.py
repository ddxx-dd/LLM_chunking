from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # 원본 (git 제외)
PROCESSED_DIR = DATA_DIR / "processed"     # processed/<dataset>/<doc_id>.json (git 제외, 다시 만들 수 있음)
QA_DIR = DATA_DIR / "qa"                   # qa/<dataset>.jsonl (git 포함)
MANIFEST_DIR = DATA_DIR / "manifests"      # manifests/<dataset>.json (git 포함)

SRT_KOR_DIR = RAW_DIR / "srt_kor"
SRT_ENG_DIR = RAW_DIR / "srt_eng"
ALLGANIZE_DIR = RAW_DIR / "allganize_pdf"  # <도메인>/<파일>.pdf
VECTARA_DIR = RAW_DIR / "vectara_pdf"      # <arXiv id>.pdf
VECTARA_CORPUS_DIR = RAW_DIR / "vectara_corpus"  # vectara 가 파싱한 섹션 텍스트 (qrels 의 section_id 기준)
DOCX_CORPUS_DIR = RAW_DIR / "docx_corpus"  # ko/, en/

RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-m3"
TOKENIZER = "google/gemma-4-12B-it-qat-q4_0-unquantized"

SUBTITLE_DATASETS = {
    "Noah": ("Noah_Eng.srt", "노아.srt"),
    "Deadpool": ("Deadpool_Eng.srt", "데드풀.srt"),
    "InsidiousChapter2": ("Insidious_Chapter2_Eng.srt", "인시디어스2.srt"),
    "DoctorStrange": ("Doctor_Strange_Eng.srt", "닥터스트레인지.srt"),
    "CaptainAmericaCivilWar": ("Captain_America_Civil_War_Eng.srt", "캡틴아메리카시빌워.srt"),
}