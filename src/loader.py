#pip install python-docx

import re
from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

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
    
    text = text.replace('\n',' ')
    
    text = re.sub(r'\s+', ' ', text)
    
    return text


#docx파일 전처리
def load_docx(filepath):
    doc = Document(filepath)
    texts = []
    for p in doc.paragraphs:
        if p.text.strip():
            texts.append(p.text.strip())
    
    return '\n'.join(texts)
          
                
            

    
    
    

    
                    
        
    
        
        
    
    