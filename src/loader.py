#pip install python-docx

import re
from docx import Document

#자막파일 전처리
def load_srt(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    
    #자막 번호 제거
    text = re.sub(r'^\d+$','',text,flags=re.MULTILINE)
    
    #시간 표시 제거
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}','',text)
    
    #빈 줄 정리
    text = re.sub(r'\n{2,}','\n',text)
    
    return text.strip()


#docx파일 전처리 (텍스트만 추출, 텍스트상자 X, 표 X)
def load_docx(filepath):
    doc = Document(filepath)
    texts = []
    for p in doc.paragraphs:
        if p.text.strip():
            texts.append(p.text.strip())
    
    return '\n'.join(texts)
    



            
    
    

    
                    
        
    
        
        
    
    