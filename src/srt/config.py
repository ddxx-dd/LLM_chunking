"""srt(자막) 데이터셋 설정. common/config.py의 DATASETS["srt"]가 이것을 그대로 쓴다. 영화 쌍은 manifests/srt.json.
smart의 lang은 번역 방향(원문 언어)마다 정한다. 이 파일은 common.config를 import하지 않는다(순환 방지)."""

SRT_DATASET = dict(
        fmt="srt", lang=("en", "ko"), raw=("srt_eng", "srt_kor"), counter="gemma",
        tasks=["translate"],
        chunkers=dict(
            fixed=dict(target_tokens=400, overlap_ratio=0.0, strip_whitespace=False),
            # semantic max는 기존 1024에서 512로 낮췄다(smart와 같은 값, 5.4절의 공정성 조정)
            semantic=dict(percentile=90, min_tokens=128, max_tokens=512, buffer_size=0),
            # srt smart는 아직 예전 방식(legacy_smart.SmartChunker). pdf·docx smart와 달리 srt 전략은 나중에 정리한다
            smart=dict(mode="srt", algorithm="legacy", min_tokens=128, max_tokens=512, window=2),
        ),
)
