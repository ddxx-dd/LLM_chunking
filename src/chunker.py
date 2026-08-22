#단순 글자수 분할 베이스라인
def fixed_chunking(text,chunk_size):
    
    if not text or not text.strip():
        return []
    
    return [text[i:i+chunk_size] for i in range(0,len(text),chunk_size)]
    
    