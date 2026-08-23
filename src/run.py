from loader import load_file
from config import SRT_ENG_DIR
from fixed_chunker import fixed_chunking


text_eng_srt = load_file(SRT_ENG_DIR/"Interstellar.srt")

fixed_chunk_list = fixed_chunking(text_eng_srt)

for i, text in enumerate(fixed_chunk_list):
    print(f"청크{i+1}: {text}")

    

    



