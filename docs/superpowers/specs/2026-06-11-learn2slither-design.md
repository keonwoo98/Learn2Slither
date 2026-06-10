# Learn2Slither — 설계 문서 (Design Spec)

날짜: 2026-06-11
대상: 42 Learn2Slither (Reinforcement Learning) — mandatory 전체 + bonus 전체

## 1. 목표와 평가 기준

서브젝트(Learn2Slither.pdf v1.00) 요구사항 요약:

### Mandatory
- **Board**: 10×10, green apple 2개(랜덤), red apple 1개(랜덤), snake 길이 3으로
  랜덤·연속(contiguous) 배치.
- **규칙**: 벽 충돌 = game over, 자기 몸 충돌 = game over,
  green apple = 길이 +1 후 새 green apple 스폰, red apple = 길이 −1 후 새 red
  apple 스폰, 길이 0 = game over.
- **State (snake vision)**: 머리에서 4방향(상/하/좌/우) ray만 볼 수 있음.
  터미널에 W/H/S/G/R/0 문자로 vision 출력 필수. **agent에 vision 외 정보 제공
  시 −42 페널티.**
- **Action**: UP, LEFT, DOWN, RIGHT 4개. 오직 vision(state) 기반으로 결정.
- **Rewards**: 자유 설계. (green +, red −, 빈 이동 작은 −, 사망 큰 − 권장)
- **Q-learning**: Q-function 기반 모델만 허용 (Q-table 또는 Neural Network).
  탐험/활용(exploration/exploitation), 반복 학습, 모델 export/import,
  학습 비활성화 스위치(dontlearn) 필수.
- **구조**: Environment / Interpreter / Agent 모듈 분리 (서브젝트 다이어그램).
- **Turn-in**: `models/` 폴더에 1, 10, 100 세션 학습 모델 최소 3개.
  학습 진행도를 보여줄 수 있어야 함.
- **CLI 예시** (서브젝트 그대로 지원해야 함):
  - `./snake -sessions 10 -save models/10sess.txt -visual off`
  - `./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step`
  - `./snake -visual on -load models/1000sess.txt`
- **출력**: 세션 종료 시 `Game over, max length = X, max duration = Y`.
- **목표 성능**: 세션 종료 시 길이 10 이상 + 긴 생존 시간.
- **일반 규칙**: 예기치 않은 크래시 금지(크래시 = 0점), Python 권장,
  flake8 norm 준수.

### Bonus
1. 세션 종료 길이 15 / 20 / 25 / 30 / 35 달성 (단계별 가점).
2. 시각적으로 뛰어난 display: lobby, 설정 패널, 결과/통계.
3. board 크기를 인자로 변경 가능 + **동일한 학습 모델이 모든 board 크기에서
   동작**해야 함.

## 2. 검토한 접근법과 결정

### 2.1 모델: Q-table vs Neural Network
- **A) Tabular Q-learning (채택)**: state 공간을 작게 인코딩하면 수천 상태
  이하 → 수 초~분 내 수만 세션 학습 가능. 결정적·재현 가능, 디펜스에서 Q-value
  를 직접 보여주며 설명 가능. 외부 의존성 불필요. board-size 보너스와 자연
  호환(상대적 feature 사용 시).
- B) DQN (torch): 표현력은 높지만 학습 시간·불안정성·디펜스 설명 비용이 크고,
  이 문제 규모에서 이득이 없음. 기각.
- C) 둘 다 구현: YAGNI. 기각. (단, Agent 인터페이스는 Q-function 추상화로
  두어 추후 확장 가능하게 함.)

### 2.2 State 인코딩 (가장 중요한 설계 결정)
제약: 머리 기준 4방향 ray에 보이는 정보만 사용 가능 (−42 페널티 조항).
ray에서 **파생된(축약된)** feature는 "보이는 정보의 부분집합"이므로 허용된다.

- **A) 방향별 binary feature (채택, 기본)**: 각 방향 d ∈ {UP, LEFT, DOWN,
  RIGHT}에 대해
  - `danger(d)`: 인접 칸(거리 1)이 W 또는 S이면 1
  - `green(d)`: ray 위에 G가 보이면 1
  - `red(d)`: ray 위에 R가 보이면 1 (green과 동일한 가시성 기준)
  → 4방향 × 3bit = 12bit ⇒ 최대 4096 states. board 크기와 무관 →
  bonus 3 충족.
- **B) 확장 feature (실험 옵션)**: A에 방향별 `body-in-ray` bit 또는 거리
  bucket(인접/근거리/원거리)을 추가해 self-trap 회피력 강화. 최대 수만 states
  로 여전히 작음. 길이 15+ bonus 달성에 필요하면 도입.
- C) ray 문자열 원본을 state key로 사용: state 폭발, board 크기 종속(bonus 3
  불가). 기각.

encoder는 전략 패턴으로 분리해 A/B를 모델 메타데이터에 기록(로드 시 자동
선택). 서브젝트의 "다른 update 전략으로 여러 모델 학습 가능" 요구와도 부합.

### 2.3 Rewards (초기값, 학습 곡선 보고 튜닝)
- green apple: **+20**
- red apple: **−20**
- 빈 이동: **−1** (배회 억제; 사망 페널티보다 충분히 작게 유지)
- game over (벽/몸/길이 0): **−100**
- shaping(사과 접근 보상)은 기본 미사용. 길이 bonus 달성이 막히면 옵션으로
  실험(환경이 계산하는 reward는 state 제약과 무관하므로 규정 위반 아님).

### 2.4 Q-learning 하이퍼파라미터 (초기값)
- 업데이트: `Q(s,a) ← Q(s,a) + α·(r + γ·max Q(s′,·) − Q(s,a))`,
  terminal이면 target = r.
- α = 0.1, γ = 0.95.
- ε-greedy: ε 1.0 → 0.01, 세션 단위 지수 감쇠(총 세션 수에 맞춰 스케줄 조정).
- Q-table: `dict[int state] → float[4]` (sparse), 미방문 state는 0 초기화.

### 2.5 기술 스택
- Python 3 (venv + `requirements.txt` 고정). **pygame–Python 버전 호환을
  구현 첫 단계에서 스모크 테스트로 확인** (현재 로컬: Python 3.14.3 +
  pygame 2.5.2 — 비호환 가능성 있어 pygame 2.6+/pygame-ce 또는 Python 3.12
  venv로 조정).
- GUI: pygame. 터미널: vision/action 텍스트 출력(서브젝트 규정).
- 테스트: pytest. 린트: flake8 (42 norm).

## 3. 아키텍처

서브젝트 다이어그램(Environment → Interpreter → Agent → Environment)을 그대로
모듈 경계로 사용한다.

```
Learn2Slither/
├── snake                    # 실행 진입점 (shebang, chmod +x) — ./snake ...
├── requirements.txt
├── README.md
├── .flake8                  # 또는 setup.cfg
├── Makefile                 # install / lint / test / train 타깃
├── models/                  # 제출용 학습 모델 (1sess.txt, 10sess.txt, 100sess.txt, best*.txt)
├── docs/
├── src/
│   ├── __init__.py
│   ├── config.py            # 상수·기본 하이퍼파라미터·reward 값
│   ├── cli.py               # argparse (-sessions, -save, -load, -visual, -dontlearn, -step-by-step, -board-size, -speed, -seed)
│   ├── environment.py       # Board/Snake: step(action) → 이벤트, 스폰, 충돌 규칙
│   ├── interpreter.py       # board → vision 문자열(터미널 출력) / state 인코딩 / reward 계산
│   ├── agent.py             # QLearningAgent: choose_action(ε-greedy), update, save/load
│   ├── display.py           # pygame GUI (visual on일 때만 import — visual off는 pygame 미설치 환경에서도 동작)
│   ├── session.py           # 학습/평가 루프 오케스트레이션, step-by-step, 통계 수집
│   └── stats.py             # max length / max duration / 학습 곡선 리포트
└── tests/
    ├── test_environment.py
    ├── test_interpreter.py
    ├── test_agent.py
    └── test_cli.py
```

### 데이터 흐름 (한 step)
1. `Environment`가 현재 board 보유.
2. `Interpreter.vision(board)` → 4방향 ray 문자열(터미널 출력) 및
   `Interpreter.state(board)` → 정수 state.
3. `Agent.choose_action(state)` → action (ε-greedy; dontlearn 시 greedy).
4. `Environment.step(action)` → 이벤트(ATE_GREEN/ATE_RED/MOVED/DIED).
5. `Interpreter.reward(event)` → r; `Agent.update(s, a, r, s′, done)`
   (dontlearn 시 생략).
6. `Display`(visual on)와 터미널 출력 갱신. step-by-step이면 입력 대기.

### 핵심 규칙 명세 (모호점 해소)
- 이동: 머리를 한 칸 전진, 꼬리 한 칸 제거. green이면 꼬리 제거 생략(+1),
  red이면 꼬리 한 칸 추가 제거(−1).
- 충돌 판정은 "이동 후 머리 칸" 기준. 몸(S)이 있는 칸으로 이동하면 game
  over(꼬리 칸 포함 — 보수적 규칙으로 통일).
- 사과 스폰: 빈 칸(뱀·다른 사과 제외) 중 균등 랜덤. 빈 칸이 없으면 스폰 보류
  (크래시 금지).
- 초기 뱀: 랜덤 위치·방향의 일직선 3칸(보드 내 보장).
- `max duration` = 해당 세션에서 살아남은 step 수, `max length` = 세션 중
  도달한 최대 길이. 세션 종료마다
  `Game over, max length = X, max duration = Y` 출력.
- 길이 1에서 red apple을 먹으면 길이 0 → game over.

### 모델 파일 포맷 (.txt 안 JSON)
```json
{
  "format_version": 1,
  "state_encoder": "binary12",
  "hyperparams": {"alpha": 0.1, "gamma": 0.95},
  "epsilon": 0.03,
  "trained_sessions": 100,
  "q_table": {"<state_int>": [q_up, q_left, q_down, q_right]}
}
```
사람이 읽을 수 있고(서브젝트 .txt 예시 부합), 인코더/세션 수 메타데이터로
재현·검증 가능. 로드 시 스키마 검증, 손상 파일은 명확한 에러 메시지로 종료
(크래시 아님).

## 4. CLI 명세

서브젝트 예시와 동일한 single-dash 플래그를 정확히 지원:

| 플래그 | 의미 | 기본값 |
|---|---|---|
| `-sessions N` | 실행할 세션 수 | 1 |
| `-save PATH` | 종료 시 모델 저장 | 없음 |
| `-load PATH` | 시작 시 모델 로드 | 없음(빈 Q-table) |
| `-visual on/off` | pygame GUI 표시 | on |
| `-dontlearn` | Q 업데이트·탐험 비활성(평가 모드) | off |
| `-step-by-step` | 키 입력마다 한 step 진행 | off |
| `-speed MS` | GUI step 간격(사람이 읽을 수 있는 속도 포함) | 150ms |
| `-board-size N` | (bonus) 보드 크기 | 10 |
| `-seed N` | 재현용 시드 | 없음 |

- 잘못된 인자/파일은 usage 메시지와 함께 정상 종료(비정상 크래시 금지).
- `-visual off` + 세션 다수 = 고속 학습 모드(터미널 출력도 옵션화).

## 5. GUI / 터미널 출력

- 터미널: 매 step, 서브젝트 그림과 동일한 십자형 vision(W/H/S/G/R/0)과 선택한
  action 출력. 학습 고속 모드에서는 생략 가능(서브젝트 허용).
- pygame: 그리드, green/red apple, snake(머리 구분), 상태바(세션 번호, 현재/
  최대 길이, step 수, ε). 키: space(step/pause), +/−(속도), q/ESC(종료).
- Bonus display: 시작 lobby(설정 패널: 세션 수, 속도, 모델 선택), 세션 종료
  결과 화면, 학습 곡선/통계 패널. mandatory 완성 후 진행.

## 6. 학습·모델 제출 전략

- 한 번의 긴 학습 실행에서 체크포인트 저장: 1, 10, 100 (+ 1000, 10000, …)
  세션 모델 → 학습 진행도가 자연스럽게 드러남 (디펜스 요구사항).
- `make train`(또는 스크립트)으로 제출 모델 일괄 재생성 가능하게 함.
- 평가 스크립트: 모델을 N게임(예: 100) `dontlearn`으로 돌려 평균/최대 길이·
  duration 분포 리포트 → mandatory(10+) 및 bonus(15~35) 달성 검증에 사용.
- bonus 길이 달성 순서: 기본 12bit encoder로 10+ 확보 → 확장 encoder·reward
  튜닝·세션 증가로 15/20/25/30/35 단계적 도전. 각 단계 결과를 기록.

## 7. 품질 / 검증

- **flake8 0 violations** (CI 수준으로 `make lint`).
- pytest 단위 테스트: 환경 규칙(충돌·성장·축소·스폰·길이 0), vision 문자열
  정확성(고정 board로 서브젝트 그림 재현), state 인코딩, Q 업데이트 수식,
  모델 save/load 라운드트립, CLI 파싱.
- 통합 테스트: seed 고정 headless 100세션 스모크(크래시 없음, 통계 출력).
- 예외 처리: 모든 입출력·인자 오류는 메시지 후 정상 종료. GUI 창 닫기 등
  중단 시그널 안전 처리.
- README: 설치(venv), 사용법, 모델 설명, 학습 재현 방법, 디펜스 체크리스트.

## 8. 리스크

| 리스크 | 대응 |
|---|---|
| pygame–Python 버전 비호환 | 구현 1단계에서 스모크 테스트, venv로 버전 고정 |
| 12bit state로 길이 15+ 한계 | 확장 encoder(B안)·reward 튜닝 준비, 단계적 측정 |
| −42 조항(정보 과다 제공) | Agent 입력을 state 정수 하나로 강제(타입 경계), 테스트로 회귀 방지 |
| 평가 환경(빈 디렉토리 clone) | requirements.txt + README 설치 절차 + `-visual off` 경로는 pygame 없이도 동작 |
| 크래시 = 0점 | 전 경로 예외 처리 + 통합 스모크 테스트 |

## 9. 구현 순서 (개요)

1. 프로젝트 스캐폴드 + venv/의존성 검증(pygame 스모크) + flake8/pytest 설정
2. Environment (보드 규칙 전체) + 테스트
3. Interpreter (vision 출력·state 인코딩·reward) + 테스트
4. Agent (Q-table, ε-greedy, update, save/load) + 테스트
5. Session 루프 + CLI (서브젝트 플래그) + 터미널 출력
6. pygame Display (speed, step-by-step)
7. 학습 실행·하이퍼파라미터 튜닝 → 모델 생성(1/10/100/…) + 평가 스크립트
8. Bonus: board-size 일반화 검증 → 길이 단계 달성 → display 고도화
9. 최종 검증: flake8, 전체 테스트, 디펜스 시나리오 리허설, README

상세 단계별 구현 계획은 별도 plan 문서로 작성한다.
