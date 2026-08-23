#pip install python-docx

import re
from docx import Document
from pathlib import Path

#자막파일 전처리
def clean_srt(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    
    #자막 번호 제거
    text = re.sub(r'^\d+$','',text,flags=re.MULTILINE)
    
    #시간 표시 제거
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}','',text)
    
    return " ".join([line.strip() for line in text.splitlines() if line.strip()])


#docx파일 전처리
def clean_docx(filepath: str) -> str:
    doc = Document(filepath)
    
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    
    return '\n'.join(texts)

def load_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    
    if ext == ".srt": 
        return clean_srt(filepath)
    elif ext == ".docx":
        return clean_docx(filepath)
    
          
                
            

    
    
    

    
                    
        
    
        
        
    
    