# Training Log

평가 방법: `python -m scripts.evaluate <model> -games 100` —
dontlearn(greedy) 모드 100게임의 max_length 도달 통계.

## Run 1 — baseline (binary12, γ0.95, αm0.1, ε 1.0→0.01 decay 0.995)

rewards: green +20 / red −20 / move −1 / death −100, seed 42

| model | mean_len | max_len | p10+ | mean_dur |
|---|---|---|---|---|
| 1sess | 3.05 | 4 | 0.00 | 3.3 |
| 10sess | 3.02 | 4 | 0.00 | 3.4 |
| 100sess | 3.23 | 5 | 0.00 | 7.7 |
| 1000sess | 15.50 | 36 | 0.81 | 129 |
| 10000sess | 19.63 | 36 | 0.90 | 167 |

mandatory(10+) 통과. bonus 상향을 위해 실험 진행.

## Run 2 — 단일 변수 실험 (각 100게임 평가)

| 실험 | mean_len | max_len | p10+ | mean_dur |
|---|---|---|---|---|
| binary16, 10k | 20.65 | 33 | 0.93 | 308 |
| binary12, 50k | 20.09 | 38 | 0.97 | 161 |
| γ0.99, 10k | 20.49 | 40 | 0.93 | 172 |

모두 소폭 개선 → 조합 실험.

## Run 3 — 조합 실험

| 실험 | mean_len | max_len | p10+ | mean_dur |
|---|---|---|---|---|
| **binary16 + γ0.99, 50k, decay 0.995** | **31.60** | **51** | **0.96** | **489** |
| binary12 + γ0.99, 50k | 19.95 | 35 | 0.95 | 184 |
| binary16 + γ0.99, 50k, decay 0.999 | 27.63 | 50 | 0.97 | 434 |

**승자: binary16 + γ0.99 + decay 0.995.** body-in-ray bit(자기 몸 인지)와
긴 보상 지평(γ0.99)의 조합이 self-trap 회피를 크게 개선.

## Run 4 — ε-decay 0.97 검증 (기각)

| 실험 | mean_len | max_len | p10+ |
|---|---|---|---|
| decay 0.97, 100sess | 3.33 | 6 | 0.00 |
| decay 0.97, 50k | 20.49 | 32 | 0.97 |

빠른 감쇠는 100sess를 거의 개선하지 못하고 50k 성능을 크게 훼손
(65k state 공간에는 긴 탐험이 필수). **decay 0.995 유지.**

## 최종 설정 (src/config.py 기본값)

binary16 encoder, α=0.1, γ=0.99, ε 1.0→0.01 (decay 0.995/session),
rewards green +20 / red −20 / move −1 / death −100,
starvation cap 500 steps.

## 최종 제출 모델 (`make train`으로 재생성, seed 42)

| model | mean_len | median | max_len | p10+ | mean_dur | max_dur |
|---|---|---|---|---|---|---|
| 1sess | 3.05 | 3 | 4 | 0.00 | 3.3 | 19 |
| 10sess | 3.08 | 3 | 5 | 0.00 | 3.8 | 13 |
| 100sess | 3.09 | 3 | 4 | 0.00 | 4.8 | 35 |
| 1000sess | 7.53 | 7 | 15 | 0.21 | 43.5 | 120 |
| 10000sess | 17.04 | 16 | 39 | 0.70 | 444 | 1236 |
| **best (50000sess)** | **31.60** | **35.5** | **51** | **0.96** | **489** | **1396** |

- mandatory(길이 10+): best 모델이 96% 게임에서 달성.
- bonus 단계: median 35.5 → **15/20/25/30/35 전 단계 달성** (max 51).
- 학습 진행도(1→10→100→1k→10k→50k)가 단조 증가 — 디펜스 시연 가능.

## Bonus: board-size 일반화 (best.txt, 50게임/크기)

| board | mean_len | max_len | p10+ |
|---|---|---|---|
| 10×10 | 31.24 | 46 | 0.96 |
| 15×15 | 48.98 | 68 | 0.98 |
| 20×20 | 60.30 | 91 | 1.00 |

상대적(relative) state 인코딩 덕분에 동일 모델이 모든 보드 크기에서
동작하며, 공간이 넓을수록 성능이 오히려 상승.

## Alternate update strategy 모델

`models/alt-binary12-50k.txt` — binary12 encoder(축소 state 공간) 학습본.
mean 19.95 / max 35 / p10+ 0.95. 재생성:
`./snake -visual off -quiet -seed 42 -sessions 50000 -encoder binary12
-save models/alt-binary12-50k.txt`
