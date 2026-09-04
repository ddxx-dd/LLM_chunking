"""
임베딩 모델 비교: bge-m3 vs Qwen3-Embedding-4B

측정 항목
  1. 모델 단독 VRAM
  2. LLM 동시 로드 시 VRAM 
  3. 인코딩 속도             -> 어블레이션 실행 횟수를 좌우
  4. 주제 전환 유사도 낙차     -> 의미 분할 정확도의 직접 지표 (핵심)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import gc
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "Qwen3-Emb-4B": "Qwen/Qwen3-Embedding-4B",
}

LLM_NAME = "Qwen/Qwen3-4B"

TEST_LLM = True

TEST_KO = [
    ("AI", [
        "머신러닝은 인공지능의 한 분야이다.",
        "딥러닝은 신경망을 여러 층으로 쌓은 구조다.",
        "신경망은 데이터로부터 패턴을 학습한다.",
    ]),
    ("주식", [
        "어제 코스피 지수가 큰 폭으로 하락했다.",
        "투자자들은 시장 변동성에 우려를 표했다.",
        "전문가들은 당분간 조정이 이어질 것으로 전망했다.",
    ]),
    ("요리", [
        "김치찌개는 한국의 대표적인 가정식이다.",
        "돼지고기와 묵은지를 함께 끓이면 깊은 맛이 난다.",
        "마지막에 두부를 넣고 살짝 더 끓인다.",
    ]),
]


TEST_EN = [
    ("Programming", [
        "A pointer stores the memory address of another variable.",
        "Arrays are contiguous blocks of memory of the same type.",
        "You can access array elements using an index.",
    ]),
    ("Weather", [
        "Heavy rain is expected across the region tomorrow.",
        "Temperatures will drop below freezing overnight.",
        "Residents are advised to stay indoors.",
    ]),
]



def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 3

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        

def flatten_sentences(test_data):
    sentences = []
    topics = []

    for topic, sentence_group in test_data:
        for sentence in sentence_group:
            sentences.append(sentence)
            topics.append(topic)

    return sentences, topics

def measure_vram(model_path):

    clear_gpu()
    before = gpu_mem_gb()

    print("  [VRAM] 임베딩 모델 로드 중...")
    model = SentenceTransformer(model_path)

    embed_vram = gpu_mem_gb() - before
    print("  [VRAM] 임베딩 모델 단독: " + format(embed_vram, ".2f") + " GB")

    total_vram = None
    both_ok = None

    if TEST_LLM:
        print("  [VRAM] LLM 동시 로드 시도 중... (시간이 걸립니다)")
        try:
            from transformers import AutoModelForCausalLM

            llm = AutoModelForCausalLM.from_pretrained(
                LLM_NAME,
                dtype=torch.float16,
                device_map="cuda",
            )

            total_vram = gpu_mem_gb() - before
            both_ok = True
            print("  [VRAM] 임베딩 + LLM 합계: "
                  + format(total_vram, ".2f") + " GB  -> 동시 로드 성공")

            del llm

        except torch.cuda.OutOfMemoryError:
            both_ok = False
            print("  [VRAM] OOM 발생 -> 동시 로드 불가")

        except Exception as e:
            both_ok = None
            print("  [VRAM] LLM 로드 실패(OOM 아님): " + str(e))

    del model
    clear_gpu()

    return {
        "embed_vram": round(embed_vram, 2),
        "total_vram": round(total_vram, 2) if total_vram else None,
        "both_ok": both_ok,
    }




def measure_speed(model, target_count=100):

    base_sentences, topics = flatten_sentences(TEST_KO)

    repeat_count = target_count // len(base_sentences) + 1
    test_sentences = (base_sentences * repeat_count)[:target_count]

    model.encode(test_sentences[:4])

    start_time = time.time()
    model.encode(test_sentences)
    elapsed = time.time() - start_time

    return round(elapsed, 2)

def measure_gap(model, test_data, label=""):

    sentences, topics = flatten_sentences(test_data)
    vectors = model.encode(sentences)

    within = []   # 같은 주제 안의 유사도
    cross = []    # 주제 경계의 유사도

    print("  [" + label + "]")

    for i in range(len(sentences) - 1):
        sim = float(cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0])

        is_boundary = topics[i] != topics[i + 1]

        if is_boundary:
            cross.append(sim)
            mark = "  <-- 주제 경계"
        else:
            within.append(sim)
            mark = ""

        print("    문장" + str(i + 1) + " <-> 문장" + str(i + 2)
              + ": " + format(sim, ".3f") + mark)

    within_avg = float(np.mean(within))
    cross_avg = float(np.mean(cross))
    gap = within_avg - cross_avg

    print("    같은 주제 평균: " + format(within_avg, ".3f"))
    print("    주제 경계 평균: " + format(cross_avg, ".3f"))
    print("    낙차(gap):      " + format(gap, ".3f"))
    print()

    return round(gap, 3)



def main():
    if torch.cuda.is_available():
        print("GPU: " + torch.cuda.get_device_name(0))
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("전체 VRAM: " + format(total, ".1f") + " GB\n")
    else:
        print("경고: GPU를 찾을 수 없습니다.\n")

    all_results = []

    for model_name, model_path in MODELS.items():
        print("=" * 60)
        print("모델: " + model_name)
        print("=" * 60)

        try:
            vram = measure_vram(model_path)

            print("  [로드] 측정용으로 다시 로드 중...")
            model = SentenceTransformer(model_path)

            speed = measure_speed(model)
            print("  [속도] 100문장 인코딩: " + str(speed) + "초\n")

            ko_gap = measure_gap(model, TEST_KO, "한국어")
            en_gap = measure_gap(model, TEST_EN, "영어")

            del model
            clear_gpu()

            all_results.append({
                "model": model_name,
                "embed_vram": vram["embed_vram"],
                "total_vram": vram["total_vram"],
                "both_ok": vram["both_ok"],
                "speed": speed,
                "ko_gap": ko_gap,
                "en_gap": en_gap,
            })

        except Exception as e:
            print("[에러] " + model_name + " 측정 실패: " + str(e) + "\n")
            clear_gpu()

    print("=" * 60)
    print("최종 비교")
    print("=" * 60)

    for r in all_results:
        print(r["model"])
        print("  단독 VRAM:      " + str(r["embed_vram"]) + " GB")
        print("  LLM 동시 로드:  " + str(r["total_vram"]) + " GB "
              + "(" + str(r["both_ok"]) + ")")
        print("  100문장 속도:   " + str(r["speed"]) + "초")
        print("  한국어 낙차:    " + str(r["ko_gap"]))
        print("  영어 낙차:      " + str(r["en_gap"]))
        print()

    print("해석")
    print("  낙차가 클수록 주제 경계를 뚜렷하게 잡음 (핵심 지표)")
    print("  속도는 낮을수록 어블레이션을 많이 돌릴 수 있음")
    print("  낙차 차이 0.05 미만이면 품질 차이가 미미하다고 볼 수 있음")


if __name__ == "__main__":
    main()
    
    