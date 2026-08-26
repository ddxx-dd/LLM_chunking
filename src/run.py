from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from loader import load_file
from config import SRT_KOR_DIR
from fixed_chunker import fixed_chunking
from semantic_chunker import semantic_chunking
from analyzer import print_chunks


#설정
FILE_PATH = SRT_KOR_DIR / "부산행.srt"
CHUNK_SIZE = 512
OVERLAP = 0
METHOD = "percentile"
AMOUNT = 10


#로드
text = load_file(FILE_PATH)
print("파일: " + FILE_PATH.name)
print("원본: " + str(len(text)) + "글자\n")

print("모델 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print("로드 완료\n")


#청킹
fixed_chunks = fixed_chunking(text,CHUNK_SIZE,OVERLAP)
semantic_chunks = semantic_chunking(text,model,method = METHOD, amount = AMOUNT)

#출력 
print_chunks(fixed_chunks,tokenizer,"단순 분할")
print_chunks(semantic_chunks,tokenizer,"의미 기반 분할")






    

    

    



