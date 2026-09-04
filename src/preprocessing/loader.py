import re
import json
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph



        


#docx파일 전처리
# 문단과 표를 문서에 나온 순서대로 내보낸다.
def _iter_body(doc):
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(child,doc)
        elif tag == "tbl":
            yield Table(child,doc)
            
# docx_표의 각 행을 JSON 한 줄 문자열로 만든다.
# -> '{"이름": "김철수", "나이": "20"}\n{"이름": "이영희", "나이": "22"}'
def _table_to_json_lines(table):
    rows_data = [[c.text.strip() for c in row.cells] for row in table.rows]
    if len(rows_data) < 2:
        return ""
    
    header = rows_data[0]
    lines = []
    
    for values in rows_data[1:]:
        row_dict = {}
        for key,val in zip(header,values):
            if key and val:
                row_dict[key] = val
        if row_dict:
            lines.append(json.dumps(row_dict, ensure_ascii=False))
    
    return "\n".join(lines)
            

def clean_docx(filepath: str) -> str:
    doc = Document(filepath)
    parts = []
    
    for item in _iter_body(doc):
        if isinstance(item, Table):
            json_lines = _table_to_json_lines(item)
            if json_lines:
                parts.append(json_lines)
        elif isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                parts.append(text)
        
    return "\n".join(parts)
        

def load_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    
    if ext == ".srt": 
        return clean_srt(filepath)
    elif ext == ".docx":
        return clean_docx(filepath)
    
    
#자막파일 전처리
def clean_srt(filepath: str) -> str:
    text = None
    for enc in ["utf-8", "cp949"]:
        try:
            with open(filepath, encoding = enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if text is None:
        raise UnicodeDecodeError("srt", b"", 0, 1, "utf-8/cp949 모두 실패")
    
    #번호 줄과 타임스탬프 줄 제거
    pattern = r'^\d+\r?\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
    text = re.sub(pattern,'',text,flags = re.MULTILINE)
    
    #자막 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    return " ".join(lines)
        
    
          

        
        
    
    
    

    
                    
        
    
        
        
    
    