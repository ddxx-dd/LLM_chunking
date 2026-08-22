from loader import load_srt, load_docx
from config import SRT_ENG_DIR, SRT_KOR_DIR, DOCX_ENG_DIR, DOCX_KOR_DIR
from chunker import fixed_chunking


text_kor_srt = load_srt(SRT_KOR_DIR/"부산행.srt")

srt_chunks = fixed_chunking(text_kor_srt,512)

for i, chunk in enumerate(srt_chunks):
    print(f"srt_chunk {i+1}: [{chunk}]")
    



