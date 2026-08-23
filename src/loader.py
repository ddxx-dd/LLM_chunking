import re
from docx import Document
from pathlib import Path

#자막파일 전처리
def clean_srt(filepath: str) -> str:
    text = ""
    encodings = ['utf-8','cp949']
    for enc in encodings:
        try:
            with open(filepath, encoding = enc) as f:
                text = f.read()
        except UnicodeDecodeError:
            continue
    
    pattern = r'^\d+\r?\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
    text = re.sub(pattern,'',text,flags = re.MULTILINE)
    
    text = re.sub(r'<[^>]+>', '', text)
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    return " ".join(lines)
        
        


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
    
          
                
            

    
    
    

    
                    
        
    
        
        
    
    