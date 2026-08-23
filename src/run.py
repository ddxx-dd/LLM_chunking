from loader import load_file
from config import SRT_KOR_DIR
from fixed_chunker import fixed_chunking

text_kor_srt = load_file(SRT_KOR_DIR/"부산행.srt")

fixed_chunk_list = fixed_chunking(text_kor_srt)

for i, text in enumerate(fixed_chunk_list):
    print(f"청크{i}: {text}")

    



