from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SRT_KOR_DIR = DATA_DIR / "srt_kor"
SRT_ENG_DIR = DATA_DIR / "srt_eng"
DOCX_KOR_DIR = DATA_DIR / "docx_kor"
DOCX_ENG_DIR = DATA_DIR / "docx_eng"
PDF_KOR_DIR = DATA_DIR / "pdf_kor"
PDF_ENG_DIR = DATA_DIR / "pdf_eng"

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