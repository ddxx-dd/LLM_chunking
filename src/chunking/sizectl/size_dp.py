"""크기 제어 DP (docs/restructure_plan.md 5.4.2절). semantic과 smart가 같이 쓰는 순수 함수 - 텍스트·임베딩·구조를 모른다.

  partition(n, size, gains, min_size, max_size) -> (그룹 [(a, b)], (min 부족분 합, 이득 합))

  n 개의 단위를 연속한 그룹으로 나눈다. 그룹 (a, b)는 단위 a..b-1이고 size(a, b)가 그 토큰 수다.
  gains[i] = 경계 i(단위 i와 i+1 사이)를 자를 때의 이득(보통 이웃 거리 - 임계값 + 구조 계층이 준 보정).
  목표는 사전식(lexicographic)이다 - 앞 단계가 엄격히 우선한다:
    1) 그룹의 크기는 max_size 이하 (하드. 단위 하나가 max를 넘는 경우만 예외로 그 단위 하나를 한 그룹으로 허용)
    2) min_size 미만 그룹의 부족분(min_size - 크기) 합을 최소화
    3) 자른 경계의 gains 합을 최대화
  그래서 min_weight 같은 튜닝 값이 없다. min을 지킬 수 있는 분할이 있으면 항상 그것이 이기고, 그 안에서 의미 경계를 가장 잘 쓴다.
  지킬 수 없는 경우(옆 단위가 max를 넘는 표 등)는 부족분이 남아 호출한 쪽이 under_min으로 보고한다.
  점화식: f[j] = max_a ( f[a] + (자른 경계 a-1의 gain) - 부족분(a..j-1) ). 복잡도 O(n*W), W = 한 그룹에 들어가는 최대 단위 수.
  같은 점수면 마지막 그룹이 짧은 쪽(a가 큰 쪽)을 고른다(결정적).
"""


def partition(n, size, gains, min_size, max_size):
    if n <= 0:
        return [], (0, 0.0)
    neg = (float("-inf"), float("-inf"))
    f, prev = [neg] * (n + 1), [0] * (n + 1)
    f[0] = (0, 0.0)
    for j in range(1, n + 1):
        for a in range(j - 1, -1, -1):
            sz = size(a, j)
            if a < j - 1 and sz > max_size:                  # 더 왼쪽으로 늘려도 더 커질 뿐이다
                break
            short = max(0, min_size - sz)
            cut = gains[a - 1] if a > 0 else 0.0
            cand = (f[a][0] - short, f[a][1] + cut)
            if cand > f[j]:
                f[j], prev[j] = cand, a
    groups, j = [], n
    while j > 0:
        groups.append((prev[j], j))
        j = prev[j]
    shortfall, gain = f[n]
    return groups[::-1], (-shortfall, gain)
