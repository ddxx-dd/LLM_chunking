import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#텍스트의 토큰 수를 센다
def count_tokens(text,tokenizer):
    return len(tokenizer.encode(text))

#청크 목록을 글자 수, 토큰 수와 함께 출력한다
#preview: 각 청크에서 보여줄 앞 부분 글자 수
def print_chunks(chunks, tokenizer, label = "",preview = 40):
    print("="*60)
    print(label)
    print("="*60)
    
    total_chars = 0
    total_tokens = 0
    
    for i in range(len(chunks)):
        chunk = chunks[i]
        chars = len(chunk)
        tokens = count_tokens(chunk,tokenizer)
        
        total_chars += chars
        total_tokens += tokens
        
        print("청크" + str(i + 1) + " | " + str(chars) + "글자 | " + str(tokens) + "토큰")
        print(chunk.strip()[:preview] + "...")
        print()
    
    print("-"*60)
    print("청크 수:       " + str(len(chunks)))
    print("평균 글자:    " + format(total_chars / len(chunks), ".1f"))
    print("평균 토큰:    " + format(total_tokens / len(chunks), ".1f"))
    print("토큰/글자 비:   " + format(total_tokens / total_chars,".2f"))
    print()
    
#인접 문장 유사도 곡선과 자른 지점을 그린다.
def plot_boundaries(similarities, threshold, save_path, title=""):
    
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    
    plt.figure(figsize = (14,5))
    
    #유사도 곡선
    plt.plot(similarities,linewidth = 0.8,color = "steelblue",label = "similarity")
    
    #임계값 수평선
    plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
                label="threshold = " + format(threshold, ".3f"))
    
    cut_x = []
    cut_y = []
    
    for i in range(len(similarities)):
        if similarities[i] < threshold:
            cut_x.append(i)
            cut_y.append(similarities[i])
            
            
    plt.scatter(cut_x,cut_y,color = "red", s=25, zorder=3, label = "cut points (" + str(len(cut_x)) + ")")
    
    plt.xlabel("sentence index")
    plt.ylabel("cosine similarity")
    plt.title = title
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi = 120)
    
    plt.close()
    
    print("저장: " + save_path)

   
