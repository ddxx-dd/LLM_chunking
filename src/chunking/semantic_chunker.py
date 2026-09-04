# 임베딩 유사도 기반 의미 분할 베이스라인

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 구두점(.!?), 줄바꿈 그리고 Json 한 줄 문자열을 기준으로 문장을 나눈다.
def split_sentences(text, max_length = 200):
    rough_pieces = re.split(r'(?<=[.!?])\s+|\n+', text)
    pieces = [p.strip() for p in rough_pieces if p.strip()]
    
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
                current = current + " " + word if current else word
        if current.strip():
            sentences.append(current.strip())
    return sentences
        


# 인접 벡터 간 코사인 유사도 계산
def calculate_similarities(vectors):
    similarities = []
    for i in range(len(vectors) - 1):
        sim_matrix = cosine_similarity([vectors[i]], [vectors[i + 1]])
        similarities.append(float(sim_matrix[0][0]))
    return similarities


# 경계 판단 기준값 계산
def calculate_threshold(similarities, method="percentile", amount=5):
    if not similarities:
        return 0.0

    if method == "fixed":
        return amount

    if method == "percentile":
        return float(np.percentile(similarities, amount))

    if method == "std":
        mean = float(np.mean(similarities))
        std = float(np.std(similarities))
        return mean - amount * std

    raise ValueError("알 수 없는 method: " + method)


# 경계 위치에서 청크 결합
def split_at_boundaries(sentences, similarities, threshold):
    chunks = []
    current = [sentences[0]]

    for i in range(len(similarities)):
        if similarities[i] < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])

    if current:
        chunks.append(" ".join(current))

    return chunks


# 의미 기반 분할 베이스라인
def semantic_chunking(text, model, method = "percentile", amount = 5, max_sentence_length=200):
    
    sentences = split_sentences(text, max_sentence_length)

    if len(sentences) < 2:
        return [text.strip()] if text.strip() else []

    vectors = model.encode(sentences)
    
    similarities = calculate_similarities(vectors)
    
    threshold = calculate_threshold(similarities, method, amount)

    return split_at_boundaries(sentences, similarities, threshold)
    