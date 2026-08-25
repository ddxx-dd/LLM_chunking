from loader import load_file
from config import DOCX_KOR_DIR
from fixed_chunker import fixed_chunking
from semantic_chunker import semantic_chunking
from sentence_transformers import SentenceTransformer


text_kor_docx = load_file(DOCX_KOR_DIR/"혼합코퍼스_A.docx")

fixed_chunk_list = fixed_chunking(text_kor_docx,512,0)

print("="*30 + "fixed_chunking"+"="*30 )
for i, text in enumerate(fixed_chunk_list):
    print(f"청크{i+1}: {text}")
print("="*60)

print("bge-m3 로드중..")
model = SentenceTransformer("BAAI/bge-m3")
print("로드 완료\n")



semantic_chunk_list = semantic_chunking(text_kor_docx,model,threshold = 0.4)

print("="*60)
for i, text in enumerate(semantic_chunk_list):
    print(f"청크{i+1}: {text}")
print("="*60)
    

    

    



