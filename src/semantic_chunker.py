import re
from sklearn.metrics.pairwise import cosine_similarity


#문장 분리
def split_sentences(text, max_length):
    
    rough_pieces = re.split(r'(?<=[.!?])\s+|\n+',text)
    
    pieces = []
    
    for piece in rough_pieces:
        clean_text = piece.strip()
        if clean_text:
            pieces.append(clean_text)
            
    sentences = []
    for piece in pieces:
        
        if len(piece) <= max_length:
            sentences.append(piece)
            continue
        
        words = piece.split(" ")
        current = ""
        
        for word in words:
            if len(current) + len(word) + 1 > max_length and current:
                sentences.append(current.strip())
                current = word
            else:
                if current:
                    current + " " + word
                else:
                    current = word
                
        
        if current.strip():
            sentences.append(current.strip())
    
        
    return sentences


#의미 기반 분할 베이스라인
def semantic_chunking(text,model,threshold = 0.5, max_sentence_length = 200):
    
    sentences = split_sentences(text,max_sentence_length)
    
    if len(sentences) < 2:
        if text.strip():
            return [text.strip()]
        else:
            return []
        
    vectors = model.encode(sentences)
    
    chunks = []
    current_sentences = []
    
    current_sentences.append(sentences[0])
    
    for i in range(1, len(sentences)):
        previous_vector = vectors[i-1]
        current_vector = vectors[i]
        
        similarity_matrix = cosine_similarity([previous_vector], [current_vector])
        similarity = float(similarity_matrix[0][0])
        
        if similarity < threshold:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])
            
    
    if current_sentences:
        chunks.append(" ".join(current_sentences))
        
    return chunks

    
    
        
        
        
    
                
    