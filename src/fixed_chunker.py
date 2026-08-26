#단순 글자수 분할 베이스라인
def fixed_chunking(text, chunk_size = 256, overlap = 0):
    
    if not text.strip():
        return []
    
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 함")
        
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start = start + step
    
    return chunks
    