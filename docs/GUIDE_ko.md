# Learn2Slither 완전 정복 가이드

> 이 문서 하나로 프로젝트 전체(개념 + 코드 + 서브젝트 요구 + 디펜스 대비)를
> 이해하는 것을 목표로 합니다. 위에서 아래로 한 번 읽으면 흐름이 이어지도록
> 구성했습니다.

---

## 목차

1. [이 프로젝트가 무엇인가](#1-이-프로젝트가-무엇인가)
2. [강화학습 기초 — 강아지 훈련 비유](#2-강화학습-기초--강아지-훈련-비유)
3. [Q-learning 핵심 — 커닝 노트](#3-q-learning-핵심--커닝-노트)
4. [전체 구조 — 7개 파일과 데이터 흐름](#4-전체-구조--7개-파일과-데이터-흐름)
5. [파일별 상세](#5-파일별-상세)
6. [한 스텝의 전체 흐름 따라가기](#6-한-스텝의-전체-흐름-따라가기)
7. [서브젝트 요구사항 ↔ 코드 매핑](#7-서브젝트-요구사항--코드-매핑)
8. [보너스 3종](#8-보너스-3종)
9. [실행 방법](#9-실행-방법)
10. [디펜스 대비 — 예상 질문과 답](#10-디펜스-대비--예상-질문과-답)
11. [용어 사전](#11-용어-사전)

---

## 1. 이 프로젝트가 무엇인가

**한 줄 요약:**

> 뱀의 머리에서 보이는 4방향 시야만 보고, **Q-learning**으로 시행착오를 거듭하며
> 사과를 먹어 길이 10 이상으로 자라는 법을 **스스로 배우는** AI를 만드는 것.

핵심은 "우리가 뱀에게 게임하는 법을 직접 코딩하지 않는다"는 점입니다. 대신
**칭찬(보상)과 혼냄(벌점)** 만 주고, 뱀이 그 점수를 최대화하는 방향으로 행동을
스스로 다듬게 합니다. 이것이 **강화학습(Reinforcement Learning)** 입니다.

만들어야 하는 것은 하나의 프로그램(`./snake`)이며, 명령어로 실행하면:

- 뱀이 보드 위에서 학습하고,
- 학습 결과(모델)를 파일로 저장/불러올 수 있고,
- 화면(GUI)과 터미널에 진행 상황을 보여줘야 합니다.

---

## 2. 강화학습 기초 — 강아지 훈련 비유

강아지에게 "앉아"를 가르칠 때 문법을 설명하지 않습니다. 앉으면 간식(보상),
엉뚱한 짓엔 간식 없음. 이걸 반복하면 강아지는 "앉으면 이득"을 **스스로**
터득합니다. 강화학습이 정확히 이것입니다.

### RL의 5가지 등장인물

| 부품 | 뜻 | 이 프로젝트에서 |
|---|---|---|
| **Environment** (환경) | 게임 세상 | 10×10 보드, 뱀, 사과, 규칙 |
| **State** (상태) | 지금 상황 | 뱀 머리의 4방향 시야 |
| **Agent** (에이전트) | 결정하는 두뇌 | 뱀의 AI |
| **Action** (행동) | 할 수 있는 선택 | UP / LEFT / DOWN / RIGHT |
| **Reward** (보상) | 칭찬/혼냄 점수 | 초록 +20, 죽음 −100 등 |

### 빙글빙글 도는 루프

강화학습은 이 다섯이 끝없이 도는 **루프**입니다(서브젝트 12쪽 다이어그램):

```
환경이 [상태]를 보여줌
   → 에이전트가 [행동]을 고름
      → 환경이 바뀌고 [보상]을 줌
         → 에이전트가 그 보상으로 배움
            → 다시 새 상태… (반복)
```

뱀은 이 루프를 **수만 번** 돌면서, "어떤 상황에서 어떤 행동을 하면 점수가
좋더라"를 몸으로 익힙니다.

---

## 3. Q-learning 핵심 — 커닝 노트

강화학습에는 여러 방법이 있는데, 이 프로젝트는 **Q-learning**(그중에서도
**Q-table** 방식)을 씁니다. 서브젝트는 Q-table 또는 신경망만 허용하며, 다른
모델을 쓰면 0점입니다.

### Q값과 Q-table = 커닝 노트

**Q값(Q-value)** 은 *"이 상황에서 이 방향으로 가면 앞으로 얼마나 좋을까?"* 를
나타내는 **예상 점수**입니다 (Q = Quality).

이 Q값들을 모은 표가 **Q-table**, 즉 **커닝 노트**입니다:

| 상황(State) | UP | LEFT | DOWN | RIGHT |
|---|---|---|---|---|
| 오른쪽에 초록 보임 | 2.1 | 0.5 | 1.0 | **15.3** ← 최고점 |
| 위가 벽 | **−48** | 5.2 | 3.1 | 4.0 |

뱀은 매 순간 "지금 상황"을 노트에서 찾아 **점수가 가장 높은 방향**으로 갑니다.
처음엔 노트가 백지(전부 0)지만, 게임을 하며 받은 보상으로 점점 채워집니다.

### 노트를 고쳐쓰는 공식 (Q-learning 업데이트)

행동하고 결과를 본 뒤, 노트의 점수를 조금씩 고쳐씁니다:

> **새 Q = 기존 Q + (학습률) × [ 받은 보상 + (할인율) × 다음 상황 최고 Q − 기존 Q ]**

겁먹을 것 없이 손잡이 두 개만 기억하면 됩니다:

- **학습률(α, alpha = 0.1):** 한 번의 경험으로 노트를 얼마나 확 바꿀까. 0.1이면
  오차의 10%만 천천히 반영(한 번의 경험은 운일 수 있으니).
- **할인율(γ, gamma = 0.99):** 당장의 보상 vs 미래의 보상 중 미래를 얼마나
  중요시할까. 0.99면 먼 미래까지 거의 똑같이 중요하게 봅니다.

**핵심 직관:** *"방금 한 행동의 가치 = 즉시 받은 보상 + 그 다음에 펼쳐질
상황의 가치"*. 좋은 결과로 이어진 선택의 점수를 끌어올리는 것입니다.

#### 마법 같은 부분: 보상의 뒤로 전파

`할인율 × 다음 상황 최고 Q` 가 보상을 **뒤로 전파**시킵니다. 사과를 먹은 칸의
가치가 오르면 → 그 옆 칸(거기로 가면 사과에 닿음)의 가치도 오르고 → 또 그 옆
칸도… 이렇게 **"사과로 가는 길" 전체의 가치가 올라가서**, 뱀이 사과가 코앞에
없어도 그쪽으로 향하게 됩니다.

### 탐험 vs 활용 (Exploration vs Exploitation)

노트의 최고점만 고집하면, 안 가본 길에 더 좋은 게 있어도 영영 못 찾습니다.
그래서 학습 중엔 가끔 **주사위를 굴려 무작위로** 움직여 봅니다.

- **활용(Exploitation):** 노트 믿고 최고점 가기 (똑똑하게)
- **탐험(Exploration):** 가끔 모험하기 (새 발견)

이 비율이 **epsilon(ε)** 입니다. 처음엔 ε=1.0(백지니까 거의 다 무작위) →
학습이 쌓이면 ε=0.01(거의 노트만 믿음)로 **판이 끝날 때마다 조금씩 줄입니다.**

---

## 4. 전체 구조 — 7개 파일과 데이터 흐름

소스는 전부 합쳐 약 767줄입니다. RL 루프가 파일에 1:1로 나뉘어 있습니다.

```
        ┌─────────────────────────────────────────┐
        │            session.py (사회자)            │
        │   "자, 한 판 시작! 다들 순서대로 일해라"    │
        └─────────────────────────────────────────┘
              │            │             │
              ▼            ▼             ▼
      environment.py  interpreter.py   agent.py
       (게임판/규칙)   (눈 + 심판)       (두뇌)
```

| 파일 | 줄 수 | 역할 | RL에서 |
|---|---|---|---|
| `config.py` | 20 | 설정값 (점수·학습률·크기) | 상수 |
| `environment.py` | 110 | 게임판·규칙 | Environment |
| `interpreter.py` | 120 | 시야→상태, 결과→보상 | State, Reward |
| `agent.py` | 95 | 두뇌 (Q-learning) | Agent |
| `session.py` | 104 | 부품들을 순서대로 호출 | 루프 |
| `cli.py` | 102 | 명령어 해석·조립·크래시 방어 | 시동 |
| `display.py` | 208 | 화면(GUI) | 보조 |

**한 스텝(뱀이 한 칸 움직이는 순간)에 도는 순서:**

1. `interpreter`가 `environment`(보드)를 **보고** → 시야를 **숫자 상태**로 만듦
2. `agent`가 그 숫자를 받아 → **방향 하나**를 고름 (노트 보고)
3. `environment`가 그 방향으로 뱀을 **움직임** → 결과 발생(먹음/죽음/이동)
4. `interpreter`가 그 결과를 → **점수(보상)** 로 환산
5. `agent`가 (상태, 행동, 점수)로 → **노트를 고쳐씀**
6. `session`이 1~5를 뱀이 죽을 때까지 **반복**, 죽으면 새 판

---

## 5. 파일별 상세

### 5.1 `config.py` — 모든 설정의 단일 출처

로직 없이 숫자만 모아둔 파일. 값이 흩어지면 튜닝이 지옥이라 한곳에 모음.

```python
# 보드 (서브젝트 Part 1)
BOARD_SIZE = 10        # 10×10
GREEN_APPLES = 2       # 초록 2개
RED_APPLES = 1         # 빨강 1개
INITIAL_LENGTH = 3     # 뱀 시작 길이 3

# 보상 (서브젝트 Part 4, 우리가 설계)
REWARD_GREEN = 20.0    # 초록 +20
REWARD_RED = -20.0     # 빨강 −20
REWARD_MOVE = -1.0     # 그냥 이동 −1 (빈둥대지 마)
REWARD_DEATH = -100.0  # 죽음 −100 (절대 안 돼)

# Q-learning 손잡이
ALPHA = 0.1            # 학습률
GAMMA = 0.99           # 할인율
EPSILON_START = 1.0    # 탐험 시작값
EPSILON_DECAY = 0.995  # 판마다 곱하는 감쇠율
EPSILON_MIN = 0.01     # 탐험 최소값

# 안전장치 / 화면
MAX_STEPS_WITHOUT_FOOD = 500  # 무한루프 방지
DEFAULT_SPEED_MS = 150        # GUI 한 스텝 0.15초
```

**보상의 핵심은 크기 순서:** `|이동 −1| < |사과 ±20| < |죽음 −100|`. 죽음이 가장
아파야 뱀이 "일단 살고 보자"를 배우고, 이동에 작은 −1을 줘서 "사과 쪽으로 빨리
가자"를 배웁니다. 만약 죽음이 −1이고 사과가 +20이면, 뱀이 "사과 먹으려고 죽음을
여러 번 감수해도 이득"이라 잘못 배웁니다.

### 5.2 `environment.py` — 게임판과 규칙

강화학습과 무관한, 순수한 "뱀 게임" 부분.

#### 기본 어휘 (Action / DELTAS / Event)

```python
class Action(IntEnum):     # 4방향. IntEnum = 이름이자 숫자
    UP = 0; LEFT = 1; DOWN = 2; RIGHT = 3

DELTAS = {                 # 방향 → (행 변화, 열 변화)
    Action.UP: (-1, 0),    # 위 = row 감소
    Action.LEFT: (0, -1),
    Action.DOWN: (1, 0),
    Action.RIGHT: (0, 1),
}

class Event(IntEnum):      # 한 스텝의 결과 4가지
    MOVED = 0; ATE_GREEN = 1; ATE_RED = 2; DIED = 3
```

- `Action`이 **숫자**인 이유: 나중에 노트의 `[UP점수, LEFT점수, DOWN점수,
  RIGHT점수]` 리스트에서 **칸 번호(인덱스)** 로 쓰기 위해. `UP=0` → 리스트 0번 칸.
- 좌표는 `(row, col)` = (세로 줄, 가로 칸). **맨 위·맨 왼쪽이 (0,0)** (바둑판처럼).
- `Event`는 environment가 "무슨 일이 일어났는지"를 알려주는 이름표. 나중에
  interpreter가 이걸 보고 보상으로 환산.

#### 게임 시작 세팅 (`__init__`, `reset`, `_place_snake`, `_spawn`)

```python
def __init__(self, size=10, rng=None):
    if size < 5: raise ValueError(...)   # 너무 작으면 거부 (크래시 대신)
    self.rng = rng or random.Random()    # 난수 발생기 (seed 주입 가능 → 재현)
    self.reset()

def reset(self):                         # 한 판 차리기
    self.snake = self._place_snake()     # 뱀 길이 3 배치
    초록 2개, 빨강 1개 _spawn
    self.alive = True; self.duration = 0; self.max_length = 3
```

- **`rng` 주입**이 핵심: `-seed`를 주면 같은 난수 → 같은 게임 전개(재현성/테스트).
- `_place_snake`: 머리를 무작위 칸에 놓고, **4방향을 섞은 뒤** 일직선 3칸이
  **보드 안에 들어가는 방향**을 찾아 배치. (머리가 구석이면 어떤 방향은 벽 밖으로
  나가니 건너뜀.) → "길이 3, 무작위, 연속" 충족.
- `_spawn`: 이미 찬 칸(뱀∪사과)을 빼고 **빈 칸에서만** 사과를 놓음. 빈 칸이
  없으면 그냥 넘어감(크래시 방지).

#### 심장 `step()` — 이동/충돌/먹기 규칙

**뱀 표현:** `self.snake`는 좌표 리스트. `snake[0]`=머리, 끝=꼬리.
**이동 트릭:** 머리 쪽에 새 칸을 추가하고, 꼬리를 몇 번 빼느냐로 길이를 조절.

```python
def step(self, action):
    self.duration += 1
    head = 현재 머리 + DELTAS[action]      # 새 머리 위치
    if 보드 밖 or head in self.snake:       # 벽/몸 충돌
        self.alive = False; return Event.DIED
    self.snake.insert(0, head)             # 머리 추가 (+1)
    if head in green_apples:               # 초록: 꼬리 안 뺌 → +1
        먹고 새 초록 스폰; event = ATE_GREEN
    elif head in red_apples:               # 빨강: 꼬리 2번 뺌 → −1
        먹고 self.snake.pop() ×2; 새 빨강 스폰
        if not self.snake: return DIED     # 길이 0 = 죽음
        event = ATE_RED
    else:                                  # 빈칸: 꼬리 1번 → 길이 유지
        self.snake.pop(); event = MOVED
    self.max_length = max(self.max_length, len(self.snake))
    return event
```

**길이 조절 한눈에:**

| 경우 | 머리 추가 | 꼬리 빼기 | 길이 변화 |
|---|---|---|---|
| 빈칸 이동 | +1 | −1 (1번) | 그대로 |
| 초록 먹음 | +1 | 0번 (안 뺌) | **+1** |
| 빨강 먹음 | +1 | −2 (2번) | **−1** |

빨강에서 꼬리를 2번 빼는 이유: 머리를 이미 +1 했으니, 순수하게 1 줄이려면 2개를
빼야 `+1 − 2 = −1`.

> **뱀은 실질적으로 3방향만 갈 수 있다** (머리 뒤=목 방향은 자기 몸이라 충돌).
> 하지만 코드는 4방향을 다 허용하고, 뒤로 가면 "죽음"으로 처리합니다. 막는 게
> 아니라 **죽음을 통해 스스로 배우게** 하는 것. (RL의 정신)

이 `step()` 하나에 서브젝트 Part 1의 게임 규칙이 전부 들어 있습니다.

### 5.3 `interpreter.py` — 눈(시야→상태) + 심판(결과→보상)

#### 시야를 글자로 (`cell_char`, `vision_lines`)

```python
def cell_char(env, cell):              # 한 칸 → 글자 하나
    if cell == env.snake[0]: return "H"   # 머리 (반드시 먼저!)
    if cell in env.snake:    return "S"   # 몸
    if cell in green_apples: return "G"
    if cell in red_apples:   return "R"
    return "0"                            # 빈칸
```

머리(H)를 몸(S)보다 **먼저** 검사해야 합니다. 머리도 `snake`에 포함되므로,
안 그러면 머리까지 S로 나옵니다.

`vision_lines`는 머리에서 **상하좌우로 끝까지** 훑어 십자 모양을 만듭니다:

```
      W       ← 위쪽 벽
      0
      G       ← 위로 보니 초록 (거리 제한 없음!)
      0
 W000H00R W   ← 가운데 가로줄, 양끝은 벽
      S       ← 아래로 보니 내 몸
      W       ← 아래쪽 벽
```

**중요한 구분:** 뱀의 시야는 인접 4칸이 아니라 **십자선 전체**(벽까지)를 봅니다.
하지만 이 "보는 것"과 "두뇌에 넣는 것(state)"은 다릅니다(아래 참고).

#### 시야를 숫자로 압축 (`_ray`, `encode_binary12/16`) — 가장 중요

```python
def _ray(env, direction):     # 한 방향으로 쭉 훑어 글자 리스트
    머리에서 한 칸씩 이동하며 글자 수집, 끝에 "W" 붙임
    # 예: 위쪽 → ['0','G','0','W']

def encode_binary12(env):     # 방향마다 3개의 예/아니오
    for 각 방향:
        ray = _ray(...)
        danger = ray[0] in ("W","S")  # 바로 앞이 벽/몸? (즉사?)
        "G" in ray                    # 이 방향에 초록 보임?
        "R" in ray                    # 이 방향에 빨강 보임?
    # 4방향 × 3질문 = 12개 O/X → 숫자 하나
```

**체크박스로 이해:**

```
방향     danger?  green?  red?
UP         X       O       X      → 0 1 0
LEFT       O       X       X      → 1 0 0
DOWN       X       X       X      → 0 0 0
RIGHT      X       X       O      → 0 0 1
```

이 12개 O(1)/X(0)를 이어붙이면 **하나의 숫자(상황 번호)**. 이게 노트(q_table)의
열쇠가 됩니다.

**왜 거리를 버리고 "있다/없다"만 담는가?** (디펜스 핵심)

1. **경우의 수를 줄여 학습이 되게:** 거리까지 담으면 상황이 거의 무한대라 같은
   상황을 두 번 만날 일이 없어 노트가 안 다져짐. "있다/없다"만 보면 최대 4096가지로
   줄어 같은 상황을 반복해 만나며 학습됨.
2. **보드 크기와 무관해짐:** "위쪽 3칸"이 아니라 "위쪽에 있음"만 보니, 8×8이든
   20×20이든 같은 상황으로 인식 → 같은 모델이 모든 크기에서 작동(**보너스 ③**).

`encode_binary16`은 여기에 방향마다 **"이 방향에 내 몸이 보이나?"** 질문을 하나
더해 16비트로 만든 것. 멀리 있는 자기 몸까지 인지해 **자기 몸으로 갇히는 실수**를
줄입니다. **우리 best 모델이 이 binary16을 쓰며, 길이를 31까지 끌어올린 결정적
요인**입니다.

`danger`만 `ray[0]`(바로 앞 한 칸)을 보고, 사과는 `in ray`(전체)를 보는
**비대칭**도 의도된 것: "다음 칸 밟으면 즉사?"는 코앞이 중요하고, 사과는 멀리
있어도 방향만 알면 되기 때문.

#### 결과를 점수로 (`reward_for`)

```python
def reward_for(event):
    ATE_GREEN → +20 / ATE_RED → −20 / DIED → −100 / 그 외(MOVED) → −1
```

`environment.step()`이 돌려준 Event를 config의 점수로 바꿔, 두뇌에게 전달하는
다리. 서브젝트 다이어그램에서 INTERPRETER가 REWARD를 만드는 화살표.

(`colorize_vision`은 터미널 가독성용. 글자는 그대로 두고 색만 입혀, 색 코드를
떼면 서브젝트 원본 형식과 동일. 진짜 터미널(TTY)에서만 색 적용 → 파이프/파일로
넘기면 순수 텍스트.)

### 5.4 `agent.py` — 두뇌 (Q-learning)

#### 노트 만들기 + 한 줄 펴보기

```python
def __init__(self, ...):
    self.q_table = {}            # 커닝 노트, 백지로 시작
    self.epsilon = 1.0           # 탐험율
    self.trained_sessions = 0

def q_values(self, state):       # 상황 번호 → 4방향 점수
    return self.q_table.setdefault(state, [0.0]*4)  # 없으면 0으로 새로 추가
```

노트는 `{상황번호: [UP, LEFT, DOWN, RIGHT 점수]}` 딕셔너리. `setdefault` 덕에
처음 보는 상황은 `[0,0,0,0]`으로 그때그때 추가됩니다(만나는 상황만 → 듬성듬성).
이 **상황번호 = interpreter가 만든 그 숫자**.

#### 행동 고르기 (정책)

```python
def choose_action(self, state, learning=True):
    if learning and 주사위 < epsilon:      # 탐험
        return 무작위 방향
    values = self.q_values(state)
    best = max(values)
    ties = [최고점인 방향들]
    return 그중 무작위                       # 활용 (동점이면 무작위)
```

- 주사위(0~1)가 epsilon보다 작으면 → 무작위(탐험), 아니면 → 최고점(활용).
- 동점이 여럿이면 그중 무작위(안 그러면 항상 첫 방향만 골라 편향).
- `learning=False`(`-dontlearn`)면 첫 if를 건너뛰어 **탐험 없이 최고점만** →
  실력 시험 모드.

> **백지 상태에서 뱀이 마구 헤매는 이유:** 노트가 비어 모든 방향이 `[0,0,0,0]`
> 동점 → 사실상 무작위 + epsilon=1.0 → 찍을 수밖에 없어 금방 죽음. 그 죽음(−100)이
> 첫 배움이 됨.

#### 노트 고쳐쓰기 (`update`) — 학습의 심장

```python
def update(self, state, action, reward, next_state, done):
    target = reward
    if not done:                            # 안 죽었으면 미래 가치 추가
        target += self.gamma * max(self.q_values(next_state))
    values = self.q_values(state)
    values[action] += self.alpha * (target - values[action])
```

- `target` = 즉시 보상 + (할인율 × 다음 상황 최고 Q) = "이 행동이 진짜 얼마나
  좋았나"의 더 나은 추정치.
- `(target - values[action])` = 오차. 그 **10%(alpha)만** 노트값을 당김.
- **죽으면(`done`) 미래가 없으니 `target = reward`만.** 안 그러면 없는 미래
  점수가 섞여 "죽음이 생각보다 안 나쁘네"로 잘못 배움.

**숫자 예시 (벽에 박아 죽음):** `reward=-100, done=True`, 기존 `values[UP]=0`
→ `target=-100` → `values[UP] += 0.1×(-100-0) = -10`. 또 겪으면 −10→−19→…
점점 −100에 수렴.

#### 판 끝 + 저장/불러오기

```python
def end_session(self):                      # 한 판 끝날 때
    self.trained_sessions += 1
    self.epsilon = max(0.01, self.epsilon * 0.995)   # 탐험 줄이기

def save(self, path):                       # 노트+상태를 JSON으로 저장
    {format_version, state_encoder, hyperparams, epsilon,
     trained_sessions, q_table} 를 파일에 기록

@classmethod
def load(cls, path, rng=None):              # 파일 읽어 두뇌 복원
    try/except로 깨진 파일은 깔끔한 ValueError (크래시 금지)
    q_table 각 줄이 4개인지 검증
```

`save`가 노트뿐 아니라 epsilon·trained_sessions·하이퍼파라미터·인코더까지
저장하는 이유: 서브젝트가 "학습 상태 **전부**를 담아 이어서 학습 가능"을
요구하기 때문. `models/`의 파일들이 이 `save`로 만들어진 것
(`1sess.txt`=1판 후, `best.txt`=5만 판 후). `load`의 try/except·검증은 평가자가
깨진 파일을 줘도 traceback 없이 깔끔히 끝내기 위한 것(크래시=0점 방어).

### 5.5 `session.py` — 사회자 (RL 루프)

직접 일하지 않고, 부품들을 순서대로 호출합니다.

```python
def run(self, sessions):
    for num in range(1, sessions+1):        # 바깥: 여러 판
        self.env.reset()
        while self.env.alive:               # 안쪽: 한 판 (죽을 때까지)
            state = encode(self.env)               # ① 눈: 시야→state
            action = agent.choose_action(state, learning)  # ② 두뇌: 행동
            event = self.env.step(action)          # ③ 게임판: 실행
            if self.learning:                      # ④ 학습 모드일 때만
                done = not self.env.alive
                next_state = state if done else encode(self.env)
                agent.update(state, action, reward_for(event), next_state, done)
            # 사과 안 먹고 500스텝이면 truncated로 중단 (무한루프 방지)
        results.append((max_length, duration))
        if self.learning: agent.end_session()      # 탐험 줄이기
        print(f"Session {num}/{sessions}: max length = ..., duration = ...")
    print(f"Game over, max length = {max_len}, max duration = {max_dur}")
```

한 줄씩 RL 루프와 매핑:

| 코드 | RL 루프 | 파일 |
|---|---|---|
| `encode(self.env)` | STATE | interpreter |
| `agent.choose_action` | ACTION | agent |
| `env.step(action)` | ENVIRONMENT | environment |
| `reward_for(event)` | REWARD | interpreter |
| `agent.update(...)` | 학습 | agent |

**갈림길 `if self.learning:`**
- 켜짐(기본): `update`로 노트 고침 + 탐험 → 똑똑해짐
- 꺼짐(`-dontlearn`): 노트 안 건드림 + 탐험 없음 → 실력만 시험

마지막 `Game over, max length = X, max duration = Y` 가 서브젝트가 요구한 출력.

### 5.6 `cli.py` — 시동 장치

#### 명령어 옵션 정의 (`build_parser`)

`argparse`로 `-sessions`, `-save`, `-load`, `-visual`, `-dontlearn`,
`-step-by-step`, `-speed`, `-board-size`, `-seed`, `-quiet`, `-encoder`를 정의.
서브젝트의 세 예시 명령이 그대로 동작합니다.

#### 부품 조립 + 실행 (`_run`)

```python
rng = random.Random(args.seed)              # 난수 하나로 전체 결정적
if args.load: agent = QLearningAgent.load(...)  # 학습된 노트 / "Load trained..."
else:         agent = QLearningAgent(...)        # 백지
env = Environment(size=args.board_size, rng=rng) # board-size 적용(보너스)
if args.visual == "on":
    from src.display import Display          # ★ visual off면 pygame 안 불러옴
    display.show_lobby([...])               # 시작 화면
runner = SessionRunner(env, agent, learning=not args.dontlearn, ...)
runner.run(args.sessions)
if args.save: agent.save(args.save)          # "Save learning state in ..."
```

- `rng` 하나를 두뇌·게임판에 함께 → `-seed`로 재현 가능.
- `from src.display import Display`가 **함수 안**에 있어서, `-visual off`면
  pygame을 아예 안 불러옴 → **pygame 없는 환경에서도 학습이 돌아감**(디펜스 안전).

#### 입구 + 안전망 (`main`)

```python
args 검증 (sessions>=1, board-size>=5, speed>=1) → 잘못되면 깔끔한 에러
try:
    _run(args)
except KeyboardInterrupt: print("Interrupted."); return 130
except Exception as exc:  print(f"snake: error: {exc}"); return 1   # ★ 크래시 방어
return 0
```

이 `try/except`가 **서브젝트 "크래시 금지"의 최후 방어선**. 어떤 에러든
traceback 대신 `snake: error: ...` 한 줄로 끝냄. (맨 위 `snake` 실행 파일 8줄은
이 `main()`을 부르는 시동 버튼.)

### 5.7 `display.py` — 화면 (보너스 ②)

#### 창 만들기 (`__init__`)

```python
self.cell = max(24, min(64, 540 // board_size))  # 셀 크기 동적 계산
self.win_w = board_px + PANEL_W(240)             # 보드 + 오른쪽 패널
self.win_h = max(board_px, MIN_WIN_H(420))       # 최소 높이 보장
```

`self.cell`을 보드 크기에 맞춰 조절해 어떤 크기든 창이 적당히 큰 540px 정도가
되게 함(작은 보드에서 창이 쪼그라드는 문제 해결).

#### 그리기 (`draw`, `_draw_board`, `_segment`, `_apple`, `_eyes`)

**원리:** 매 스텝 화면을 싹 지우고(`fill`) 처음부터 다시 그린 뒤 `flip()`으로
한꺼번에 표시(더블 버퍼링). 보드가 100칸이라 매번 다시 그려도 비용이 사실상 0
→ 코드가 단순하고 버그 없음(이 규모에선 표준 방식).

```python
def draw(self, env, info_lines):
    self.screen.fill(BG)            # 지우기
    self._draw_board(env)           # 격자 + 사과(원) + 뱀(둥근 사각) + 머리 눈
    self._draw_side_panel(info)     # 오른쪽 통계 패널
    pygame.display.flip()           # 표시
```

- 좌표 변환: `x = col × cell`, `y = row × cell` (col이 가로, row가 세로).
- 뱀=둥근 사각형, 사과=원, 머리=`_head_dir`(머리−목 벡터)로 방향 잡아 눈 2개.

#### 속도·키 입력 (`tick`, `_pump`, `_wait_for_step`)

```python
def tick(self, step_by_step):       # 매 스텝 호출
    if step_by_step or paused: return self._wait_for_step()  # 키 대기
    speed_ms 동안 _pump로 키 받으며 대기                       # 자동 진행
```

- `_pump`: 게임 중 실시간 키 — 창닫기/ESC/Q(종료), +/−(속도), P(일시정지).
- `_wait_for_step`: 키 누를 때까지 멈춤 — SPACE(한 스텝), P(재개), Q(종료).
  `-step-by-step`과 일시정지에서 사용.

#### lobby / 결과 화면 (`show_lobby`, `show_summary`) — 보너스 ②

- `show_lobby`: 시작 화면(설정 요약) → SPACE로 시작, Q로 종료.
- `show_summary`: 결과 화면(세션/최대 길이/평균 등 통계) → 아무 키나 누르면 닫힘.

---

## 6. 한 스텝의 전체 흐름 따라가기

뱀이 한 칸 움직이는 순간, 7개 파일이 어떻게 협력하는지 끝까지:

```
[session.run의 while 루프 안]

1. state = encode(env)
   └ interpreter._ray로 4방향 훑기 → encode_binary16으로 16비트 숫자
   └ (environment의 snake/apple 위치를 cell_char로 읽음)

2. (verbose면) interpreter.vision_lines + colorize_vision → 터미널에 십자 출력

3. action = agent.choose_action(state, learning)
   └ q_table에서 state의 4방향 점수 펴봄
   └ 주사위 < epsilon? → 무작위(탐험) / 아니면 → 최고점(활용)

4. event = environment.step(action)
   └ DELTAS[action]으로 새 머리 계산
   └ 벽/몸? → DIED / 초록? → +1 ATE_GREEN / 빨강? → −1 ATE_RED / 빈칸? → MOVED

5. (learning이면) agent.update(state, action, reward_for(event), next_state, done)
   └ interpreter.reward_for(event)로 점수 환산
   └ target = 보상 + 0.99 × 다음 상황 최고 Q (죽었으면 보상만)
   └ q_table[state][action]을 target 쪽으로 10% 당김

6. display.draw(env, info) + tick → 화면 갱신 + 속도 조절(+ 키 입력)

[죽으면 while 종료 → agent.end_session()(탐험 감쇠) → 다음 판 reset]
```

이 한 스텝이 **5만 판 × 한 판당 수백 스텝 = 수백만 번** 반복되며 노트가 완성됩니다.

---

## 7. 서브젝트 요구사항 ↔ 코드 매핑

| 서브젝트 | 요구 | 코드 위치 |
|---|---|---|
| Part 1 보드 | 10×10, 사과, 뱀 배치, 충돌/먹기 규칙 | `environment.py` (`reset`, `step`) |
| Part 1 GUI | 보드 표시, 속도 조절, step 모드 | `display.py`, `cli.py` |
| Part 2 State | 4방향 시야 터미널 출력 | `interpreter.vision_lines` |
| Part 2 −42 | agent엔 시야만 (정수 state 하나) | `session.py` (state만 전달) |
| Part 3 Action | UP/LEFT/DOWN/RIGHT + 터미널 표시 | `environment.Action`, `session.py` |
| Part 4 Rewards | 보상 설계 | `config.py`, `interpreter.reward_for` |
| Part 5 Q-learning | Q-table, 탐험, 저장/불러오기, dontlearn | `agent.py` |
| Part 6 구조 | Env/Interpreter/Agent 분리 | 파일 분리 자체 |
| 제출 | models/ 폴더, 1·10·100 세션 모델 | `models/*.txt` |
| 제출 | 명령어 형식, 출력 문구 | `cli.py`, `session.py` |
| 일반 | 크래시 금지, flake8 | `cli.main` try/except, 전체 |

**−42 조항 방어의 핵심:** agent가 받는 것은 오직 `choose_action(state)`의 정수
state 하나뿐. environment 객체나 좌표·길이·전체 보드는 절대 전달되지 않음.
state는 4방향 ray에서만 파생됨.

---

## 8. 보너스 3종

보너스는 따로 떨어진 코드가 아니라 본 코드에 녹아 있습니다.

1. **더 긴 길이 (15~35):** `agent.py`(Q-learning) + `config.py`(보상)의 결과.
   best 모델은 100게임 평균 길이 ~31, 중앙값 ~35, 최대 51. 평가표 기준 만점권.
2. **멋진 화면:** `display.py`의 lobby(시작) + summary(결과/통계) + 사이드 패널
   (실시간 통계).
3. **보드 크기 변경:** `cli.py`의 `-board-size` + `interpreter.py`의 거리를 버린
   state 인코딩. **같은 모델 파일이 8×8~20×20에서 모두 작동.** (평가표는 8×8~15×15
   에서 최소 길이 7 요구 → 모든 크기 97~100% 달성.)

---

## 9. 실행 방법

```bash
# 설치 (최초 1회)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 백지 뱀 (학습 안 됨) — 금방 죽음
./snake -visual on -sessions 3 -dontlearn

# 학습된 고수 뱀
./snake -visual on -sessions 3 -dontlearn -load models/best.txt

# 직접 학습시켜 모델 만들기 (화면 끄고 빠르게)
./snake -sessions 100 -visual off -save models/my.txt

# 모델 성능 평가 (통계)
python -m scripts.evaluate models/best.txt -games 100

# 품질 확인
make lint   # flake8
make test   # pytest
```

**GUI 키:** SPACE(시작/한 스텝), P(일시정지/재개), +/−(속도), Q/ESC(종료).

---

## 10. 디펜스 대비 — 예상 질문과 답

평가표에는 "코드를 설명하지 못하면 감점(Can't support/explain code)"이 있습니다.
아래는 자주 나오는 질문과 핵심 답변.

**Q. Q값이 뭐고 어떻게 계산하나?**
A. "이 상황에서 이 방향이 앞으로 얼마나 좋은지"의 예상 점수. 업데이트는
`Q += α(보상 + γ·다음상황 최고Q − Q)`. α=0.1(천천히), γ=0.99(미래 중시).
죽으면 미래가 없으니 보상만 사용. (`agent.update`)

**Q. 보상을 왜 이렇게 정했나?**
A. 크기 순서 `|이동| < |사과| < |죽음|`이 핵심. 죽음이 가장 아파야 생존을 배우고,
이동에 작은 −1을 줘 사과로 빨리 향하게. (`config.py`)

**Q. agent에 시야만 준다는 걸 코드로 증명해봐라. (−42 조항)**
A. `session.py`에서 `agent.choose_action(state)`로 정수 state 하나만 전달.
state는 `interpreter._ray`(머리 4방향)에서만 파생. environment 객체·좌표·길이는
전혀 안 넘어감.

**Q. state를 왜 거리 없이 "있다/없다"로만 만드나?**
A. (1) 경우의 수를 4096개로 줄여 학습이 되게, (2) 보드 크기와 무관해져 같은
모델이 모든 크기에서 작동(보너스 ③).

**Q. 탐험은 어디서 일어나나?**
A. `choose_action`에서 주사위 < epsilon이면 무작위. epsilon은 판마다 0.995배로
감소(`end_session`), 최소 0.01.

**Q. dontlearn은 뭘 하나?**
A. `session.py`의 `if self.learning:`을 꺼서 update·탐험·epsilon 감쇠를 모두 생략.
노트를 안 건드리고 실력만 시험. 모델 검증용.

**Q. 학습 진행을 어떻게 보여주나?**
A. `models/`에 1·10·100·1000·10000·50000 세션 모델. `scripts.evaluate`로 평가하면
세션이 많을수록 평균 길이가 단조 증가(3 → 7 → 17 → 31).

**Q. 크래시는 어떻게 막나?**
A. `cli.main`의 `try/except`가 모든 에러를 `snake: error: ...` 한 줄로. 모델
로드도 검증 후 깔끔한 ValueError. 잘못된 인자는 argparse가 usage로 처리.

**Q. 같은 모델이 다른 보드 크기에서 되는 이유?**
A. state가 절대 좌표가 아니라 머리 기준 상대적 ray feature라 보드 크기에
의존하지 않음. `-board-size`로 크기만 바꿔 같은 `q_table`을 그대로 사용.

---

## 11. 용어 사전

| 용어 | 쉬운 뜻 |
|---|---|
| 강화학습(RL) | 칭찬/벌점만 주고 스스로 배우게 하는 방식 |
| Agent (에이전트) | 결정하는 두뇌 (뱀의 AI) |
| Environment (환경) | 게임 세상 (보드·규칙) |
| State (상태) | 지금 상황을 요약한 숫자 |
| Action (행동) | 4방향 선택지 |
| Reward (보상) | 행동에 매겨지는 점수 |
| Q값 (Q-value) | "이 상황 이 방향"의 예상 점수 |
| Q-table | 상황별 4방향 점수표 (커닝 노트) |
| Q-learning | Q-table을 보상으로 고쳐가며 배우는 알고리즘 |
| 학습률 (α) | 한 경험을 노트에 얼마나 반영할지 (0.1) |
| 할인율 (γ) | 미래 보상을 얼마나 중요시할지 (0.99) |
| 탐험 (exploration) | 가끔 무작위로 모험 |
| 활용 (exploitation) | 노트 믿고 최고점 선택 |
| epsilon (ε) | 탐험할 확률 (1.0 → 0.01) |
| 에피소드/세션 | 한 판 (뱀 태어나서 죽을 때까지) |
| 인코딩 (encode) | 시야를 숫자 state로 압축 |
| dontlearn | 학습 끄고 실력만 시험하는 모드 |

---

*이 문서는 코드(`src/`)와 함께 읽으면 가장 효과적입니다. 막히는 부분은 해당 파일의
실제 코드를 열어 이 문서의 설명과 짝지어 보세요.*
