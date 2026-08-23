#단순 글자수 분할 베이스라인
def fixed_chunking(text: str, chunk_size: int = 256 , overlap: int = 0):
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size-overlap)
        
        if start >= text_len or chunk_size <= overlap:
            break
            
    return chunks
    
    