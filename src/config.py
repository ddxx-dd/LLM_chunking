from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SRT_KOR_DIR = DATA_DIR / "srt_kor"
SRT_ENG_DIR = DATA_DIR / "srt_eng"
DOCX_KOR_DIR = DATA_DIR / "docx_kor"
DOCX_ENG_DIR = DATA_DIR / "docx_eng"
RESULTS_DIR = ROOT / "results"