import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#문장 분리
def split_sentences(text, max_length=200):
    
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
                    current = current + " " + word
                else:
                    current = word
                
        
        if current.strip():
            sentences.append(current.strip())
    
        
    return sentences

#인접한 벡터간 코사인 유사도를 모두 계산 후 리스트로 변환
#similarities[i] = 문장 i와 문장 i+1 사이의 유사도
def calculate_similarities(vectors):
    
    similarities = []
    
    for i in range(len(vectors) - 1):
        sim_matrix = cosine_similarity([vectors[i]],[vectors[i+1]])
        similarities.append(float(sim_matrix[0][0]))
    
    return similarities

#유사도 리스트를 보고 경계 판단 기준값을 계산한다.

#    method="fixed"      : amount를 그대로 임계값으로 사용(문서마다 튜닝필요)
#    method="percentile" : 유사도 하위 amount%를 경계로(문서에 자동 적응하지만 경계가 없어도 강제 비율만큼 자름)
#    method="std"        : 평균 - (amount*표준 편차)를 경계로. 경계가 없으면 안 자를 수도 있으나 극단값에 평균과 표준편차가 왜곡될 수 있다.

def calculate_threshold(similarities,method ="percentile",amount = 5):
    if method == "fixed":
        return amount
    
    if method == "percentile":
        return float(np.percentile(similarities,amount))
    
    if method == "std":
        mean = float(np.mean(similarities))
        std = float(np.std(similarities))
        return mean-amount*std
    
    raise ValueError("알 수 없는 method: "+method)
        



#유사도 리스트를 보고 경계를 찾아 청크로 묶는다
#similarities[i]가 threshold보다 낮으면 문장 i와 i+1 사이에서 자른다.
def split_at_boundaries(sentences,similarities,threshold):
    
    chunks = []
    current = [sentences[0]]
    
    for i in range(len(similarities)):
        if similarities[i] < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i+1]]
        else:
            current.append(sentences[i+1])
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


#의미 기반 분할 베이스라인
def semantic_chunking(text,model,method = "percentile",amount = 5,buffer_size = 0, max_sentence_length = 200):

    #문장분리
    sentences = split_sentences(text,max_sentence_length)
    
    if len(sentences) < 2:
        if text.strip():
            return [text.strip()]
        else:
            return []
        
    #임베딩
    buffered = combine_with_buffer(sentences,buffer_size)
    vectors = model.encode(buffered)
    
    #유사도 계산
    similarities = calculate_similarities(vectors)
    
    #임계값 계산
    threshold = calculate_threshold(similarities,method,amount)
    
    #경계에서 분할
    return split_at_boundaries(sentences,similarities,threshold)



#각 문장에 앞 뒤 문장을 붙여 문맥을 확장한다.
#buffer_size = 1이면 문장 2기준 문장1,문장3 결합
#문장들이 겹쳐지기 때문에 평균 유사도는 올라간다.
def combine_with_buffer(sentences, buffer_size = 1):
    
    if buffer_size <= 0:
        return sentences
    
    combined = []
    
    #기준 문장 인덱스
    for i in range(len(sentences)):
        parts = []
        
        for j in range(i-buffer_size, i + buffer_size + 1):
            #첫/마지막 문장 처리
            if 0 <= j < len(sentences):
                parts.append(sentences[j])
            
        combined.append(" ".join(parts))
    
    return combined
    
    

    
    
        
        
        
    
                
    