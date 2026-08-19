from loader import load_srt
from config import SRT_KOR_DIR

srt_text = load_srt(SRT_KOR_DIR/"the_truman_show.srt")
print(srt_text[:200])

