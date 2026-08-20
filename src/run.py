from loader import load_srt, load_docx
from config import SRT_ENG_DIR, SRT_KOR_DIR, DOCX_ENG_DIR, DOCX_KOR_DIR

srt_kor_text = load_srt(SRT_KOR_DIR/"트루먼쇼.srt")
print(srt_kor_text[:200])
print(len(srt_kor_text))

srt_eng_text = load_srt(SRT_ENG_DIR/"About_Time.srt")
print(srt_eng_text[:200])
print(len(srt_eng_text))

docx_kor_text = load_docx(DOCX_KOR_DIR/"배열.docx")
print(docx_kor_text[:200])
print(len(docx_kor_text))

docx_eng_text = load_docx(DOCX_ENG_DIR/"function.docx")
print(docx_eng_text[:200])
print(len(docx_eng_text))
