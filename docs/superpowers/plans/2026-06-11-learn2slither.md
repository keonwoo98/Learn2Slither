# Learn2Slither Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 42 Learn2Slither — Q-learning으로 학습하는 snake agent를 mandatory 전체 + bonus 전체(길이 단계, board-size 일반화, display 고도화) 기준으로 구현한다.

**Architecture:** 서브젝트 다이어그램 그대로 Environment(보드 규칙) / Interpreter(vision·state·reward) / Agent(Q-table) / Display(pygame) / Session(루프) / CLI 모듈로 분리. State는 4방향 ray에서 파생한 board-size 무관 binary feature(기본 12bit)로 인코딩하여 Q-table을 작게 유지하고 board-size bonus와 호환시킨다.

**Tech Stack:** Python 3 (venv), pygame(GUI), pytest(테스트), flake8(42 norm). 외부 ML 라이브러리 불필요(순수 tabular Q-learning).

**설계 문서:** `docs/superpowers/specs/2026-06-11-learn2slither-design.md`

> **주의(사용자 전역 규칙):** 커밋 단계는 사용자가 실행을 승인한 경우에만 수행한다. 절대 push하지 않는다.

---

## 파일 구조 (최종)

```
Learn2Slither/
├── snake                    # 실행 진입점 (chmod +x)
├── requirements.txt
├── README.md
├── .flake8
├── .gitignore
├── Makefile
├── pytest.ini
├── conftest.py              # pytest가 repo root를 sys.path에 넣도록 함
├── models/                  # 제출용 학습 모델 (git 추적 필수!)
├── scripts/
│   └── evaluate.py          # 모델 성능 통계 리포트
├── src/
│   ├── __init__.py
│   ├── config.py            # 상수·하이퍼파라미터·reward
│   ├── environment.py       # Action/Event/Environment(보드 규칙)
│   ├── interpreter.py       # vision 출력, state encoder, reward
│   ├── agent.py             # QLearningAgent (ε-greedy, update, save/load)
│   ├── session.py           # SessionRunner (학습/평가 루프)
│   ├── display.py           # pygame GUI
│   └── cli.py               # argparse + main()
└── tests/
    ├── test_environment.py
    ├── test_interpreter.py
    ├── test_agent.py
    ├── test_session.py
    └── test_cli.py
```

---

### Task 1: 프로젝트 스캐폴드 & 툴체인 검증

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.flake8`, `pytest.ini`, `conftest.py`, `Makefile`, `src/__init__.py`, `models/.gitkeep`

- [ ] **Step 1: venv 생성 및 의존성 설치 + pygame 호환성 스모크 테스트**

```bash
cd /Users/keokim/42/Learn2Slither
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install pygame flake8 pytest
venv/bin/python -c "import pygame; pygame.init(); print(pygame.ver)"
```

Expected: pygame 버전 출력 (예: `2.6.x`).
**실패 시(현재 로컬 Python 3.14.3에서 pygame wheel 미지원 가능):**
`venv/bin/pip install pygame-ce` 로 대체(import 이름 동일하게 `pygame`).
그래도 실패하면 사용 가능한 구버전 파이썬으로 venv 재생성:
`python3.12 -m venv venv` (또는 `brew list | grep python`으로 확인).

- [ ] **Step 2: requirements.txt 작성 (설치 성공한 버전으로 고정)**

```bash
venv/bin/pip freeze | grep -iE "^(pygame|flake8|pytest)" > requirements.txt
cat requirements.txt
```

Expected: 3개 패키지가 버전과 함께 기록됨.

- [ ] **Step 3: 설정 파일 작성**

`.gitignore`:
```
venv/
__pycache__/
*.pyc
.pytest_cache/
```
(주의: `models/`는 제출 대상이므로 절대 ignore하지 않는다.)

`.flake8`:
```
[flake8]
exclude = .git,__pycache__,venv,.venv
```

`pytest.ini`:
```
[pytest]
testpaths = tests
```

`conftest.py` (빈 파일 — pytest가 repo root를 `sys.path`에 추가하게 함):
```python
```

`src/__init__.py` (빈 파일), `models/.gitkeep` (빈 파일), `tests/` 디렉토리 생성.

- [ ] **Step 4: Makefile 작성**

```make
PYTHON = venv/bin/python

install:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

lint:
	venv/bin/flake8 src tests scripts

test:
	venv/bin/pytest -q

train:
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 1 \
		-save models/1sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 10 \
		-save models/10sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 100 \
		-save models/100sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 1000 \
		-save models/1000sess.txt
	$(PYTHON) snake -visual off -quiet -seed 42 -sessions 10000 \
		-save models/10000sess.txt

.PHONY: install lint test train
```

- [ ] **Step 5: 검증**

```bash
venv/bin/flake8 src
venv/bin/pytest -q
```

Expected: flake8 출력 없음(클린). pytest는 "no tests ran" (exit code 5 —
이 시점에서는 정상).

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt .flake8 pytest.ini conftest.py \
    Makefile src/__init__.py models/.gitkeep
git commit -m "chore: project scaffold with venv, flake8, pytest, Makefile"
```

---

### Task 2: config + Action/Event 기본 타입

**Files:**
- Create: `src/config.py`, `src/environment.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_environment.py`:
```python
from src.environment import Action, DELTAS, Event


def test_actions_are_up_left_down_right():
    assert [a.name for a in Action] == ["UP", "LEFT", "DOWN", "RIGHT"]


def test_deltas_move_one_cell():
    assert DELTAS[Action.UP] == (-1, 0)
    assert DELTAS[Action.LEFT] == (0, -1)
    assert DELTAS[Action.DOWN] == (1, 0)
    assert DELTAS[Action.RIGHT] == (0, 1)


def test_events_exist():
    assert {e.name for e in Event} == {
        "MOVED", "ATE_GREEN", "ATE_RED", "DIED"}
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_environment.py -v`
Expected: FAIL — `ModuleNotFoundError` 또는 `ImportError`.

- [ ] **Step 3: 구현**

`src/config.py`:
```python
"""Default constants, hyperparameters and rewards for Learn2Slither."""

BOARD_SIZE = 10
GREEN_APPLES = 2
RED_APPLES = 1
INITIAL_LENGTH = 3

REWARD_GREEN = 20.0
REWARD_RED = -20.0
REWARD_MOVE = -1.0
REWARD_DEATH = -100.0

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

MAX_STEPS_WITHOUT_FOOD = 500
DEFAULT_SPEED_MS = 150
```

`src/environment.py`:
```python
"""Snake board environment: subject rules, spawning, step logic."""
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


DELTAS = {
    Action.UP: (-1, 0),
    Action.LEFT: (0, -1),
    Action.DOWN: (1, 0),
    Action.RIGHT: (0, 1),
}


class Event(IntEnum):
    MOVED = 0
    ATE_GREEN = 1
    ATE_RED = 2
    DIED = 3
```
(`import random`과 `from src import config`는 Task 3에서 처음 사용되므로
이 시점에는 넣지 않는다 — flake8 F401 방지.)

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_environment.py -v && venv/bin/flake8 src tests`
Expected: 3 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/environment.py tests/test_environment.py
git commit -m "feat: add config constants and Action/Event types"
```

---

### Task 3: Environment 초기화 (snake 배치, apple 스폰)

**Files:**
- Modify: `src/environment.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_environment.py`에 추가:
```python
import random

from src.environment import Environment


def make_env(seed=42, size=10):
    return Environment(size=size, rng=random.Random(seed))


def test_initial_board_setup():
    env = make_env()
    assert env.size == 10
    assert len(env.snake) == 3
    assert len(env.green_apples) == 2
    assert len(env.red_apples) == 1
    assert env.alive is True
    assert env.duration == 0
    assert env.max_length == 3


def test_snake_is_contiguous_straight_and_in_bounds():
    for seed in range(50):
        env = make_env(seed=seed)
        cells = env.snake
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        # 일직선: 한 축은 고정, 다른 축은 연속 3칸
        assert len(rows) == 1 or len(cols) == 1
        varying = sorted(cols) if len(rows) == 1 else sorted(rows)
        assert varying[2] - varying[0] == 2 and len(varying) == 3
        for r, c in cells:
            assert 0 <= r < env.size and 0 <= c < env.size


def test_no_overlap_between_snake_and_apples():
    for seed in range(50):
        env = make_env(seed=seed)
        body = set(env.snake)
        apples = env.green_apples | env.red_apples
        assert not body & apples
        assert len(env.green_apples | env.red_apples) == 3


def test_spawn_uses_only_free_cells():
    env = make_env()
    # 한 칸만 남기고 모두 뱀으로 채움
    env.snake = [(r, c) for r in range(10) for c in range(10)
                 if (r, c) != (0, 0)]
    env.green_apples = set()
    env.red_apples = set()
    env._spawn(env.green_apples)
    assert env.green_apples == {(0, 0)}
    # 빈 칸이 없으면 스폰 보류(크래시 금지)
    env._spawn(env.red_apples)
    assert env.red_apples == set()


def test_board_too_small_rejected():
    try:
        Environment(size=4)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_environment.py -v`
Expected: 새 테스트들 FAIL — `Environment` 미정의.

- [ ] **Step 3: 구현**

`src/environment.py`의 Event 아래에 추가. 파일 상단 import를 다음으로
교체:
```python
import random
from enum import IntEnum

from src import config
```
클래스 추가:
```python
class Environment:
    """Snake board of size x size cells, following the subject rules."""

    def __init__(self, size=config.BOARD_SIZE, rng=None):
        if size < 5:
            raise ValueError("board size must be >= 5")
        self.size = size
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    def reset(self):
        """Start a new game: snake of 3, two green apples, one red."""
        self.snake = self._place_snake()
        self.green_apples = set()
        self.red_apples = set()
        for _ in range(config.GREEN_APPLES):
            self._spawn(self.green_apples)
        for _ in range(config.RED_APPLES):
            self._spawn(self.red_apples)
        self.alive = True
        self.duration = 0
        self.max_length = len(self.snake)

    @property
    def length(self):
        return len(self.snake)

    def _in_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.size and 0 <= col < self.size

    def _place_snake(self):
        head = (self.rng.randrange(self.size),
                self.rng.randrange(self.size))
        directions = list(DELTAS.values())
        self.rng.shuffle(directions)
        for d_row, d_col in directions:
            cells = [(head[0] + i * d_row, head[1] + i * d_col)
                     for i in range(config.INITIAL_LENGTH)]
            if all(self._in_bounds(cell) for cell in cells):
                return cells
        raise RuntimeError("board too small to place the snake")

    def _spawn(self, apples):
        occupied = set(self.snake) | self.green_apples | self.red_apples
        free = [(r, c)
                for r in range(self.size)
                for c in range(self.size)
                if (r, c) not in occupied]
        if free:
            apples.add(self.rng.choice(free))
```

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_environment.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: environment init with snake placement and apple spawn"
```

---

### Task 4: Environment.step — 이동/성장/축소/사망 규칙

**Files:**
- Modify: `src/environment.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_environment.py`에 추가:
```python
def fixed_env(snake, green=(), red=(), size=10, seed=0):
    """테스트용: 결정된 배치로 환경을 구성한다."""
    env = make_env(seed=seed, size=size)
    env.snake = list(snake)
    env.green_apples = set(green)
    env.red_apples = set(red)
    env.max_length = len(env.snake)
    return env


def test_plain_move_keeps_length():
    env = fixed_env([(5, 5), (5, 4), (5, 3)])
    event = env.step(Action.RIGHT)
    assert event == Event.MOVED
    assert env.snake == [(5, 6), (5, 5), (5, 4)]
    assert env.alive and env.duration == 1


def test_wall_collision_dies():
    env = fixed_env([(0, 5), (1, 5), (2, 5)])
    event = env.step(Action.UP)
    assert event == Event.DIED
    assert env.alive is False


def test_body_collision_dies():
    # 머리 (5,5), 몸이 오른쪽→아래로 꺾여 (5,6),(4,6),(4,5)
    env = fixed_env([(5, 5), (5, 6), (4, 6), (4, 5)])
    event = env.step(Action.UP)  # (4,5)는 몸
    assert event == Event.DIED
    assert env.alive is False


def test_green_apple_grows_and_respawns():
    env = fixed_env([(5, 5), (5, 4), (5, 3)], green=[(5, 6), (0, 0)])
    event = env.step(Action.RIGHT)
    assert event == Event.ATE_GREEN
    assert env.length == 4
    assert env.snake[0] == (5, 6)
    assert len(env.green_apples) == 2  # 새 green apple 스폰
    assert (5, 6) not in env.green_apples
    assert env.max_length == 4


def test_red_apple_shrinks_and_respawns():
    env = fixed_env([(5, 5), (5, 4), (5, 3)], red=[(5, 6)])
    event = env.step(Action.RIGHT)
    assert event == Event.ATE_RED
    assert env.length == 2
    assert env.snake == [(5, 6), (5, 5)]
    assert len(env.red_apples) == 1  # 새 red apple 스폰
    assert env.alive


def test_red_apple_at_length_one_is_game_over():
    env = fixed_env([(5, 5)], red=[(5, 6)])
    event = env.step(Action.RIGHT)
    assert event == Event.DIED
    assert env.alive is False
    assert env.length == 0


def test_step_after_death_raises():
    env = fixed_env([(0, 5), (1, 5), (2, 5)])
    env.step(Action.UP)
    try:
        env.step(Action.DOWN)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_duration_counts_steps():
    env = fixed_env([(5, 5), (5, 4), (5, 3)])
    env.step(Action.RIGHT)
    env.step(Action.RIGHT)
    assert env.duration == 2
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_environment.py -v`
Expected: 새 테스트 FAIL — `step` 미정의 (`AttributeError`).

- [ ] **Step 3: 구현**

`src/environment.py`의 `Environment`에 추가:
```python
    def step(self, action):
        """Advance one step in the given direction. Returns the Event."""
        if not self.alive:
            raise RuntimeError("step() called on a finished game")
        self.duration += 1
        d_row, d_col = DELTAS[action]
        head = (self.snake[0][0] + d_row, self.snake[0][1] + d_col)
        if not self._in_bounds(head) or head in self.snake:
            self.alive = False
            return Event.DIED
        self.snake.insert(0, head)
        if head in self.green_apples:
            self.green_apples.discard(head)
            self._spawn(self.green_apples)
            event = Event.ATE_GREEN
        elif head in self.red_apples:
            self.red_apples.discard(head)
            self.snake.pop()
            self.snake.pop()
            self._spawn(self.red_apples)
            if not self.snake:
                self.alive = False
                return Event.DIED
            event = Event.ATE_RED
        else:
            self.snake.pop()
            event = Event.MOVED
        self.max_length = max(self.max_length, len(self.snake))
        return event
```

규칙 메모(설계 문서 합치):
- 충돌 판정은 이동 후 머리 칸 기준, 꼬리 칸 포함(보수적 규칙).
- red apple: 머리 전진 후 꼬리 2칸 제거 = 순길이 −1. 길이 0이면 DIED.
- 스폰은 사과를 먹은 즉시(빈 칸 없으면 보류).

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_environment.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: environment step rules (move, grow, shrink, deaths)"
```

---

### Task 5: Interpreter — vision 터미널 출력 (서브젝트 그림 재현)

**Files:**
- Create: `src/interpreter.py`
- Test: `tests/test_interpreter.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_interpreter.py`:
```python
import random

from src.environment import Environment
from src.interpreter import cell_char, vision_lines


def fixed_env(snake, green=(), red=(), size=10):
    env = Environment(size=size, rng=random.Random(0))
    env.snake = list(snake)
    env.green_apples = set(green)
    env.red_apples = set(red)
    return env


def test_cell_chars():
    env = fixed_env([(5, 5), (5, 4)], green=[(1, 1)], red=[(2, 2)])
    assert cell_char(env, (5, 5)) == "H"
    assert cell_char(env, (5, 4)) == "S"
    assert cell_char(env, (1, 1)) == "G"
    assert cell_char(env, (2, 2)) == "R"
    assert cell_char(env, (9, 9)) == "0"


def test_vision_matches_subject_figure():
    # 서브젝트 IV.2 그림과 동일한 보드 구성:
    # head (7,9), body (8,9),(8,8) / green (2,9),(3,6) / red (3,9)
    env = fixed_env(
        [(7, 9), (8, 9), (8, 8)],
        green=[(2, 9), (3, 6)],
        red=[(3, 9)],
    )
    indent = " " * 10
    assert vision_lines(env) == [
        indent + "W",
        indent + "0",
        indent + "0",
        indent + "G",
        indent + "R",
        indent + "0",
        indent + "0",
        indent + "0",
        "W000000000HW",
        indent + "S",
        indent + "0",
        indent + "W",
    ]
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_interpreter.py -v`
Expected: FAIL — `src.interpreter` 미존재.

- [ ] **Step 3: 구현**

`src/interpreter.py`:
```python
"""Interpreter: board -> terminal vision, state encoding and rewards."""


def cell_char(env, cell):
    """Single character for a board cell (subject IV.2 notation)."""
    if cell == env.snake[0]:
        return "H"
    if cell in env.snake:
        return "S"
    if cell in env.green_apples:
        return "G"
    if cell in env.red_apples:
        return "R"
    return "0"


def vision_lines(env):
    """Cross-shaped 4-direction vision lines, as in the subject figure."""
    head_row, head_col = env.snake[0]
    indent = " " * (head_col + 1)
    lines = [indent + "W"]
    for row in range(head_row):
        lines.append(indent + cell_char(env, (row, head_col)))
    middle = "W" + "".join(
        cell_char(env, (head_row, col))
        for col in range(env.size)) + "W"
    lines.append(middle)
    for row in range(head_row + 1, env.size):
        lines.append(indent + cell_char(env, (row, head_col)))
    lines.append(indent + "W")
    return lines
```
(이 시점에는 import가 필요 없다. `config`/`DELTAS`/`Action`/`Event`는
Task 6에서 추가한다 — flake8 F401 방지.)

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_interpreter.py -v && venv/bin/flake8 src tests`
Expected: PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter.py tests/test_interpreter.py
git commit -m "feat: terminal vision output matching the subject figure"
```

---

### Task 6: Interpreter — state encoder(binary12/binary16) + rewards

**Files:**
- Modify: `src/interpreter.py`
- Test: `tests/test_interpreter.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_interpreter.py`에 추가:
```python
from src import config
from src.environment import Event
from src.interpreter import ENCODERS, encode_binary12, reward_for


def test_state_encodes_adjacent_danger():
    # 머리 (0,0): UP 벽, LEFT 벽, DOWN 몸, RIGHT 빈칸
    env = fixed_env([(0, 0), (1, 0), (2, 0)])
    state = encode_binary12(env)
    # bit 배치: 방향 i(UP=0,LEFT=1,DOWN=2,RIGHT=3) * 3 + {danger,green,red}
    assert state == (1 << 0) | (1 << 3) | (1 << 6)


def test_state_sees_apples_along_rays():
    env = fixed_env([(5, 5), (6, 5), (7, 5)],
                    green=[(5, 9)], red=[(0, 5)])
    state = encode_binary12(env)
    expected = (
        (1 << 2)        # UP: red 보임
        | (1 << 6)      # DOWN: 몸 인접 danger
        | (1 << 10)     # RIGHT: green 보임
    )
    assert state == expected


def test_state_is_board_size_independent():
    # 같은 상대 배치라면 10x10과 20x20에서 동일한 state (bonus 근거)
    env_small = fixed_env([(5, 5), (6, 5), (7, 5)], green=[(5, 8)])
    env_large = fixed_env([(9, 9), (10, 9), (11, 9)], green=[(9, 12)],
                          size=20)
    assert encode_binary12(env_small) == encode_binary12(env_large)


def test_encoder_registry():
    assert set(ENCODERS) == {"binary12", "binary16"}
    env = fixed_env([(5, 5), (6, 5), (7, 5)])
    for encode in ENCODERS.values():
        assert isinstance(encode(env), int)


def test_rewards():
    assert reward_for(Event.ATE_GREEN) == config.REWARD_GREEN
    assert reward_for(Event.ATE_RED) == config.REWARD_RED
    assert reward_for(Event.MOVED) == config.REWARD_MOVE
    assert reward_for(Event.DIED) == config.REWARD_DEATH
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_interpreter.py -v`
Expected: 새 테스트 FAIL — `ENCODERS` 미정의.

- [ ] **Step 3: 구현**

`src/interpreter.py` 파일 상단(docstring 아래)에 import 추가:
```python
from src import config
from src.environment import DELTAS, Action, Event
```
함수 추가:
```python
def _ray(env, direction):
    """Cell chars from head outward (head excluded), wall-terminated."""
    d_row, d_col = DELTAS[direction]
    row, col = env.snake[0]
    row, col = row + d_row, col + d_col
    chars = []
    while 0 <= row < env.size and 0 <= col < env.size:
        chars.append(cell_char(env, (row, col)))
        row, col = row + d_row, col + d_col
    chars.append("W")
    return chars


def encode_binary12(env):
    """3 bits per direction: danger adjacent, green seen, red seen."""
    state = 0
    for i, direction in enumerate(Action):
        ray = _ray(env, direction)
        danger = ray[0] in ("W", "S")
        state |= (
            (danger << (3 * i))
            | (("G" in ray) << (3 * i + 1))
            | (("R" in ray) << (3 * i + 2))
        )
    return state


def encode_binary16(env):
    """binary12 + 'snake body seen along ray' bit per direction."""
    state = 0
    for i, direction in enumerate(Action):
        ray = _ray(env, direction)
        danger = ray[0] in ("W", "S")
        state |= (
            (danger << (4 * i))
            | (("G" in ray) << (4 * i + 1))
            | (("R" in ray) << (4 * i + 2))
            | (("S" in ray) << (4 * i + 3))
        )
    return state


ENCODERS = {
    "binary12": encode_binary12,
    "binary16": encode_binary16,
}


def reward_for(event):
    """Reward granted by the board for the event of the last action."""
    if event == Event.ATE_GREEN:
        return config.REWARD_GREEN
    if event == Event.ATE_RED:
        return config.REWARD_RED
    if event == Event.DIED:
        return config.REWARD_DEATH
    return config.REWARD_MOVE
```

**−42 페널티 방지 핵심:** agent에는 이 정수 state 하나만 전달된다.
state는 4방향 ray(vision)에서만 파생 — vision 외 정보 사용 금지 회귀는
`test_state_is_board_size_independent` 등으로 방지.

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_interpreter.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter.py tests/test_interpreter.py
git commit -m "feat: vision-derived state encoders and reward mapping"
```

---

### Task 7: Agent — ε-greedy 정책, Q-update, ε 감쇠

**Files:**
- Create: `src/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_agent.py`:
```python
import random

from src import config
from src.agent import QLearningAgent
from src.environment import Action


def make_agent(epsilon=0.0, seed=0):
    return QLearningAgent(epsilon=epsilon, rng=random.Random(seed))


def test_greedy_picks_best_action():
    agent = make_agent(epsilon=0.0)
    agent.q_table[7] = [1.0, 5.0, 2.0, 3.0]
    assert agent.choose_action(7) == Action.LEFT


def test_unknown_state_initialized_to_zero():
    agent = make_agent()
    assert agent.q_values(123) == [0.0, 0.0, 0.0, 0.0]


def test_exploration_returns_valid_action():
    agent = make_agent(epsilon=1.0)
    for _ in range(20):
        assert agent.choose_action(0) in list(Action)


def test_learning_false_is_pure_greedy_even_with_high_epsilon():
    agent = make_agent(epsilon=1.0)
    agent.q_table[7] = [0.0, 0.0, 9.0, 0.0]
    for _ in range(20):
        assert agent.choose_action(7, learning=False) == Action.DOWN


def test_update_terminal_moves_toward_reward():
    agent = make_agent()
    agent.update(5, Action.UP, 10.0, 0, done=True)
    # Q = 0 + alpha * (10 - 0) = 1.0
    assert agent.q_table[5][Action.UP] == 1.0


def test_update_bootstraps_next_state():
    agent = make_agent()
    agent.q_table[2] = [0.0, 4.0, 0.0, 0.0]
    agent.update(1, Action.RIGHT, 1.0, 2, done=False)
    # target = 1 + 0.95*4 = 4.8 ; Q = 0 + 0.1*4.8 = 0.48
    assert abs(agent.q_table[1][Action.RIGHT] - 0.48) < 1e-9


def test_epsilon_decays_to_floor():
    agent = make_agent(epsilon=config.EPSILON_MIN * 1.001)
    agent.end_session()
    agent.end_session()
    assert agent.epsilon == config.EPSILON_MIN
    assert agent.trained_sessions == 2
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_agent.py -v`
Expected: FAIL — `src.agent` 미존재.

- [ ] **Step 3: 구현**

`src/agent.py`:
```python
"""Q-learning agent: sparse Q-table with an epsilon-greedy policy."""
import random

from src import config
from src.environment import Action


class QLearningAgent:
    """Tabular Q-learning. The only input it ever gets is the state."""

    FORMAT_VERSION = 1

    def __init__(self, alpha=config.ALPHA, gamma=config.GAMMA,
                 epsilon=config.EPSILON_START,
                 encoder_name="binary12", rng=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.encoder_name = encoder_name
        self.rng = rng if rng is not None else random.Random()
        self.q_table = {}
        self.trained_sessions = 0

    def q_values(self, state):
        return self.q_table.setdefault(state, [0.0] * len(Action))

    def choose_action(self, state, learning=True):
        """Epsilon-greedy when learning; pure greedy otherwise."""
        if learning and self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))
        values = self.q_values(state)
        best = max(values)
        ties = [a for a in Action if values[a] == best]
        return self.rng.choice(ties)

    def update(self, state, action, reward, next_state, done):
        """One Q-learning backup for the (s, a, r, s') transition."""
        target = reward
        if not done:
            target += self.gamma * max(self.q_values(next_state))
        values = self.q_values(state)
        values[action] += self.alpha * (target - values[action])

    def end_session(self):
        """Decay exploration once per finished training session."""
        self.trained_sessions += 1
        self.epsilon = max(config.EPSILON_MIN,
                           self.epsilon * config.EPSILON_DECAY)
```
(`json` import는 Task 8의 save/load에서 사용 — 이 시점에는 넣지 않는다.)

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_agent.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: tabular Q-learning agent with epsilon-greedy policy"
```

---

### Task 8: Agent — 모델 save/load (export/import)

**Files:**
- Modify: `src/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_agent.py`에 추가:
```python
def test_save_load_roundtrip(tmp_path):
    agent = make_agent(epsilon=0.3)
    agent.alpha = 0.2
    agent.gamma = 0.9
    agent.encoder_name = "binary16"
    agent.trained_sessions = 42
    agent.q_table[5] = [1.0, 2.0, 3.0, 4.0]
    path = str(tmp_path / "model.txt")
    agent.save(path)
    loaded = QLearningAgent.load(path)
    assert loaded.alpha == 0.2
    assert loaded.gamma == 0.9
    assert loaded.epsilon == 0.3
    assert loaded.encoder_name == "binary16"
    assert loaded.trained_sessions == 42
    assert loaded.q_table == {5: [1.0, 2.0, 3.0, 4.0]}


def test_load_rejects_invalid_file(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a model")
    try:
        QLearningAgent.load(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_rejects_wrong_version(tmp_path):
    path = tmp_path / "old.txt"
    path.write_text('{"format_version": 99}')
    try:
        QLearningAgent.load(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_agent.py -v`
Expected: 새 테스트 FAIL — `save` 미정의.

- [ ] **Step 3: 구현**

`src/agent.py` 상단에 `import json` 추가, 클래스에 메서드 추가:
```python
    def save(self, path):
        """Export the full learning state to a human-readable file."""
        data = {
            "format_version": self.FORMAT_VERSION,
            "state_encoder": self.encoder_name,
            "hyperparams": {"alpha": self.alpha, "gamma": self.gamma},
            "epsilon": self.epsilon,
            "trained_sessions": self.trained_sessions,
            "q_table": {str(s): v for s, v in self.q_table.items()},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    @classmethod
    def load(cls, path, rng=None):
        """Import a learning state previously written by save()."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not a valid model file: {path}") from exc
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {path}")
        agent = cls(
            alpha=data["hyperparams"]["alpha"],
            gamma=data["hyperparams"]["gamma"],
            epsilon=data["epsilon"],
            encoder_name=data["state_encoder"],
            rng=rng,
        )
        agent.trained_sessions = data["trained_sessions"]
        agent.q_table = {
            int(state): [float(v) for v in values]
            for state, values in data["q_table"].items()
        }
        return agent
```

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_agent.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: model export/import with schema validation"
```

---

### Task 9: SessionRunner — 학습/평가 루프

**Files:**
- Create: `src/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_session.py`:
```python
import random

from src.agent import QLearningAgent
from src.environment import Action, Environment
from src.session import SessionRunner


def test_training_run_returns_one_result_per_session(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(rng=rng)
    runner = SessionRunner(env, agent, verbose=False, log_sessions=False)
    results = runner.run(5)
    assert len(results) == 5
    for length, duration in results:
        assert length >= 0 and duration > 0
    assert agent.trained_sessions == 5
    assert agent.q_table  # 학습이 일어나 Q-table이 채워짐
    out = capsys.readouterr().out
    assert "Game over, max length =" in out
    assert "max duration =" in out


def test_dontlearn_does_not_touch_agent(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(epsilon=0.5, rng=rng)
    runner = SessionRunner(env, agent, learning=False,
                           verbose=False, log_sessions=False)
    runner.run(3)
    assert agent.trained_sessions == 0
    assert agent.epsilon == 0.5
    assert all(v == [0.0] * 4 for v in agent.q_table.values())


def test_verbose_prints_vision_and_action(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(rng=rng)
    runner = SessionRunner(env, agent, verbose=True, log_sessions=False)
    runner.run(1)
    out = capsys.readouterr().out
    assert "H" in out and "W" in out  # vision 출력
    assert any(a.name in out for a in Action)  # action 출력


class _CycleAgent:
    """2x2 사각형을 영원히 도는 스텁 — starvation cap 검증용."""

    encoder_name = "binary12"
    epsilon = 0.0

    def __init__(self):
        self._cycle = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        self._i = 0

    def choose_action(self, state, learning=True):
        action = self._cycle[self._i % 4]
        self._i += 1
        return action

    def update(self, *args):
        pass

    def end_session(self):
        pass


class _LoopEnv(Environment):
    """reset이 항상 같은 배치를 만드는 테스트용 환경."""

    def reset(self):
        super().reset()
        self.snake = [(0, 0)]
        self.green_apples = {(9, 9), (9, 8)}
        self.red_apples = {(8, 9)}
        self.alive = True
        self.duration = 0
        self.max_length = 1


def test_starvation_cap_ends_looping_session(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "MAX_STEPS_WITHOUT_FOOD", 20)
    env = _LoopEnv(rng=random.Random(0))
    runner = SessionRunner(env, _CycleAgent(), learning=False,
                           verbose=False, log_sessions=False)
    results = runner.run(1)
    assert results[0][1] == 20  # 20 스텝에서 강제 종료
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_session.py -v`
Expected: FAIL — `src.session` 미존재.

- [ ] **Step 3: 구현**

`src/session.py`:
```python
"""Training / evaluation loop orchestration."""

from src import config
from src.environment import Event
from src.interpreter import ENCODERS, reward_for, vision_lines


class SessionRunner:
    """Runs sessions: env <-> interpreter <-> agent, plus I/O pacing."""

    def __init__(self, env, agent, learning=True, step_by_step=False,
                 verbose=True, display=None, log_sessions=True):
        self.env = env
        self.agent = agent
        self.learning = learning
        self.step_by_step = step_by_step
        self.verbose = verbose
        self.display = display
        self.log_sessions = log_sessions

    def run(self, sessions):
        """Run all sessions. Returns a list of (max_length, duration)."""
        encode = ENCODERS[self.agent.encoder_name]
        results = []
        stopped = False
        for num in range(1, sessions + 1):
            self.env.reset()
            steps_since_food = 0
            while self.env.alive:
                if not self._render(num):
                    stopped = True
                    break
                state = encode(self.env)
                if self.verbose:
                    print("\n".join(vision_lines(self.env)))
                action = self.agent.choose_action(state, self.learning)
                if self.verbose:
                    print(action.name)
                event = self.env.step(action)
                if self.learning:
                    done = not self.env.alive
                    next_state = state if done else encode(self.env)
                    self.agent.update(state, action, reward_for(event),
                                      next_state, done)
                if event in (Event.ATE_GREEN, Event.ATE_RED):
                    steps_since_food = 0
                else:
                    steps_since_food += 1
                if steps_since_food >= config.MAX_STEPS_WITHOUT_FOOD:
                    break
            results.append((self.env.max_length, self.env.duration))
            if self.learning:
                self.agent.end_session()
            if self.log_sessions:
                print(f"Session {num}/{sessions}: "
                      f"length = {self.env.max_length}, "
                      f"duration = {self.env.duration}")
            if stopped:
                break
        max_len = max((r[0] for r in results), default=0)
        max_dur = max((r[1] for r in results), default=0)
        print(f"Game over, max length = {max_len}, "
              f"max duration = {max_dur}")
        return results

    def _render(self, session_num):
        """Draw the board and pace the loop. False means user quit."""
        if self.display is None:
            return True
        info = (f"session {session_num}  length {self.env.length}  "
                f"eps {self.agent.epsilon:.2f}")
        self.display.draw(self.env, info)
        return self.display.tick(self.step_by_step)
```

메모:
- `MAX_STEPS_WITHOUT_FOOD`(기본 500)은 greedy 정책의 무한 순환 방지용
  안전장치 — 사과를 먹으면 리셋된다. 학습 페널티는 주지 않는다(이동당
  `REWARD_MOVE`가 이미 배회를 억제).
- dontlearn 모드: `learning=False` → 탐험 없음, update 없음, ε 불변
  (서브젝트의 "exploitation without learning").

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest tests/test_session.py -v && venv/bin/flake8 src tests`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add src/session.py tests/test_session.py
git commit -m "feat: session runner with training and evaluation modes"
```

---

### Task 10: CLI + `snake` 진입점 + 통합 스모크

**Files:**
- Create: `src/cli.py`, `snake`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli.py`:
```python
import json

import pytest

from src.cli import build_parser, main


def test_subject_flags_parse():
    args = build_parser().parse_args(
        ["-sessions", "10", "-save", "models/10sess.txt",
         "-visual", "off"])
    assert args.sessions == 10
    assert args.save == "models/10sess.txt"
    assert args.visual == "off"


def test_subject_eval_flags_parse():
    args = build_parser().parse_args(
        ["-visual", "on", "-load", "models/100sess.txt",
         "-sessions", "10", "-dontlearn", "-step-by-step"])
    assert args.load == "models/100sess.txt"
    assert args.dontlearn is True
    assert args.step_by_step is True
    assert args.visual == "on"


def test_defaults():
    args = build_parser().parse_args([])
    assert args.sessions == 1
    assert args.visual == "on"
    assert args.board_size == 10
    assert args.dontlearn is False


def test_invalid_sessions_exits():
    with pytest.raises(SystemExit):
        main(["-sessions", "0", "-visual", "off"])


def test_train_and_save_then_load_and_eval(tmp_path):
    model = str(tmp_path / "m.txt")
    code = main(["-sessions", "3", "-visual", "off", "-quiet",
                 "-seed", "7", "-save", model])
    assert code == 0
    with open(model, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["trained_sessions"] == 3
    code = main(["-sessions", "2", "-visual", "off", "-quiet",
                 "-seed", "7", "-load", model, "-dontlearn"])
    assert code == 0


def test_missing_model_file_is_graceful_error(capsys):
    code = main(["-load", "/nonexistent/model.txt", "-visual", "off"])
    assert code == 1
    assert "error" in capsys.readouterr().err
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL — `src.cli` 미존재.

- [ ] **Step 3: 구현**

`src/cli.py`:
```python
"""Command-line interface using the subject's single-dash flags."""
import argparse
import random
import sys

from src import config
from src.agent import QLearningAgent
from src.environment import Environment
from src.interpreter import ENCODERS
from src.session import SessionRunner


def build_parser():
    parser = argparse.ArgumentParser(
        prog="snake",
        description="Learn2Slither: a snake that learns by Q-learning")
    parser.add_argument("-sessions", type=int, default=1,
                        help="number of sessions to run")
    parser.add_argument("-save", metavar="PATH",
                        help="save the learning state when done")
    parser.add_argument("-load", metavar="PATH",
                        help="load a learning state before running")
    parser.add_argument("-visual", choices=["on", "off"], default="on",
                        help="show the pygame board window")
    parser.add_argument("-dontlearn", action="store_true",
                        help="evaluation mode: no exploration/updates")
    parser.add_argument("-step-by-step", dest="step_by_step",
                        action="store_true",
                        help="advance one step per SPACE key press")
    parser.add_argument("-speed", type=int,
                        default=config.DEFAULT_SPEED_MS,
                        help="display delay per step (ms)")
    parser.add_argument("-board-size", dest="board_size", type=int,
                        default=config.BOARD_SIZE,
                        help="board size (bonus)")
    parser.add_argument("-seed", type=int, help="random seed")
    parser.add_argument("-quiet", action="store_true",
                        help="suppress per-step terminal output")
    parser.add_argument("-encoder", choices=sorted(ENCODERS),
                        default="binary12",
                        help="state encoder for a fresh agent")
    return parser


def _run(args):
    rng = random.Random(args.seed)
    if args.load:
        agent = QLearningAgent.load(args.load, rng=rng)
        print(f"Load trained model from {args.load}")
    else:
        agent = QLearningAgent(encoder_name=args.encoder, rng=rng)
    if agent.encoder_name not in ENCODERS:
        raise ValueError(
            f"unknown state encoder '{agent.encoder_name}' in model")
    env = Environment(size=args.board_size, rng=rng)
    display = None
    try:
        if args.visual == "on":
            from src.display import Display
            display = Display(args.board_size, args.speed)
        runner = SessionRunner(
            env, agent,
            learning=not args.dontlearn,
            step_by_step=args.step_by_step,
            verbose=not args.quiet,
            display=display)
        runner.run(args.sessions)
    finally:
        if display is not None:
            display.close()
    if args.save:
        agent.save(args.save)
        print(f"Save learning state in {args.save}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sessions < 1:
        parser.error("-sessions must be >= 1")
    if args.board_size < 5:
        parser.error("-board-size must be >= 5")
    if args.speed < 1:
        parser.error("-speed must be >= 1")
    try:
        _run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # 서브젝트 규칙: 예기치 않은 크래시 금지
        print(f"snake: error: {exc}", file=sys.stderr)
        return 1
    return 0
```

`snake` (repo root, 실행 파일):
```python
#!/usr/bin/env python3
"""Learn2Slither entry point: ./snake [options]"""
import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

```bash
chmod +x snake
```

- [ ] **Step 4: 통과 확인 + 수동 스모크**

```bash
venv/bin/pytest -q && venv/bin/flake8 src tests
venv/bin/python snake -sessions 2 -visual off -seed 1
```

Expected: 테스트 전체 PASS, flake8 클린. 스모크 실행에서 vision(십자형
W/H/S/G/R/0)과 action 이름이 출력되고 마지막에
`Game over, max length = X, max duration = Y`가 출력됨.

- [ ] **Step 5: Commit**

```bash
git add src/cli.py snake tests/test_cli.py
git commit -m "feat: CLI with subject flags and executable entry point"
```

---

### Task 11: pygame Display + step-by-step/speed 연결

**Files:**
- Create: `src/display.py`

(pygame 창은 자동 테스트가 어려우므로 이 Task는 구현 + 수동 검증.
검증 체크리스트를 빠짐없이 수행한다.)

- [ ] **Step 1: 구현**

`src/display.py`:
```python
"""Pygame board display: speed control, pause and step-by-step mode."""
import pygame

CELL = 40
MARGIN = 1
PANEL_H = 40
BG = (24, 24, 24)
GRID = (52, 52, 52)
GREEN = (40, 200, 80)
RED = (220, 60, 60)
BODY = (60, 110, 220)
HEAD = (130, 180, 255)
TEXT = (235, 235, 235)


class Display:
    """Dedicated window for the board (subject IV.3 info box)."""

    def __init__(self, board_size, speed_ms):
        pygame.init()
        side = board_size * CELL
        self.speed_ms = speed_ms
        self.paused = False
        self.screen = pygame.display.set_mode((side, side + PANEL_H))
        pygame.display.set_caption("Learn2Slither")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

    def draw(self, env, info=""):
        self.screen.fill(BG)
        for row in range(env.size):
            for col in range(env.size):
                self._cell(row, col, GRID)
        for row, col in env.green_apples:
            self._cell(row, col, GREEN)
        for row, col in env.red_apples:
            self._cell(row, col, RED)
        for i, (row, col) in enumerate(env.snake):
            self._cell(row, col, HEAD if i == 0 else BODY)
        label = self.font.render(info, True, TEXT)
        self.screen.blit(label, (8, env.size * CELL + 10))
        pygame.display.flip()

    def _cell(self, row, col, color):
        rect = pygame.Rect(col * CELL + MARGIN, row * CELL + MARGIN,
                           CELL - 2 * MARGIN, CELL - 2 * MARGIN)
        pygame.draw.rect(self.screen, color, rect)

    def tick(self, step_by_step):
        """Pace one step and process input. False means user quit."""
        if step_by_step or self.paused:
            return self._wait_for_step()
        deadline = pygame.time.get_ticks() + self.speed_ms
        while pygame.time.get_ticks() < deadline:
            if not self._pump():
                return False
            self.clock.tick(120)
        return True

    def _pump(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_ms = max(10, self.speed_ms - 25)
                elif event.key == pygame.K_MINUS:
                    self.speed_ms = min(1000, self.speed_ms + 25)
                elif event.key == pygame.K_p:
                    self.paused = True
        return True

    def _wait_for_step(self):
        """Wait for SPACE (one step) or P (resume). False on quit."""
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
                if event.key == pygame.K_p:
                    self.paused = False
                    return True

    def close(self):
        pygame.quit()
```

- [ ] **Step 2: lint + 기존 테스트 회귀 확인**

Run: `venv/bin/flake8 src tests && venv/bin/pytest -q`
Expected: 클린, 전체 PASS (display는 visual on일 때만 import되므로
헤드리스 테스트에 영향 없음).

- [ ] **Step 3: 수동 검증 체크리스트**

```bash
venv/bin/python snake -sessions 2 -visual on -seed 1 -speed 150
```
- [ ] 창에 그리드/green 2개/red 1개/뱀(머리 색 구분)이 보인다
- [ ] 터미널에 vision과 action이 step마다 출력된다
- [ ] `+`/`-` 키로 속도가 변한다, `p`로 일시정지/재개된다
- [ ] `q` 또는 창 닫기 시 즉시 정상 종료(트레이스백 없음)

```bash
venv/bin/python snake -sessions 1 -visual on -seed 1 -step-by-step
```
- [ ] SPACE를 누를 때만 한 step씩 진행된다

- [ ] **Step 4: Commit**

```bash
git add src/display.py
git commit -m "feat: pygame display with speed, pause and step-by-step"
```

---

### Task 12: 학습 파이프라인 — evaluate 스크립트

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/test_cli.py` (통합 1건 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_cli.py`에 추가:
```python
def test_evaluate_script_reports_stats(tmp_path, capsys):
    model = str(tmp_path / "m.txt")
    main(["-sessions", "5", "-visual", "off", "-quiet",
          "-seed", "3", "-save", model])
    capsys.readouterr()
    from scripts.evaluate import evaluate
    stats = evaluate(model, games=5, board_size=10, seed=1)
    assert stats["games"] == 5
    assert stats["max_length"] >= 3
    assert stats["mean_duration"] > 0
```

- [ ] **Step 2: 실패 확인**

Run: `venv/bin/pytest tests/test_cli.py -v`
Expected: 새 테스트 FAIL — `scripts.evaluate` 미존재.

- [ ] **Step 3: 구현**

`scripts/__init__.py` (빈 파일) 생성, `scripts/evaluate.py`:
```python
#!/usr/bin/env python3
"""Evaluate a trained model over many games and print statistics."""
import argparse
import random
import statistics

from src.agent import QLearningAgent
from src.environment import Environment
from src.session import SessionRunner


def evaluate(model_path, games=100, board_size=10, seed=42):
    """Run games in dontlearn mode; return aggregate statistics."""
    rng = random.Random(seed)
    agent = QLearningAgent.load(model_path, rng=rng)
    env = Environment(size=board_size, rng=rng)
    runner = SessionRunner(env, agent, learning=False,
                           verbose=False, log_sessions=False)
    results = runner.run(games)
    lengths = [r[0] for r in results]
    durations = [r[1] for r in results]
    return {
        "games": len(results),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "max_length": max(lengths),
        "p10_plus_rate": sum(x >= 10 for x in lengths) / len(lengths),
        "mean_duration": statistics.mean(durations),
        "max_duration": max(durations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("-games", type=int, default=100)
    parser.add_argument("-board-size", dest="board_size", type=int,
                        default=10)
    parser.add_argument("-seed", type=int, default=42)
    args = parser.parse_args()
    stats = evaluate(args.model, args.games, args.board_size, args.seed)
    print(f"model: {args.model}  board: {args.board_size}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

실행은 repo root에서 `venv/bin/python -m scripts.evaluate models/X.txt`
(모듈 실행이라 `sys.path` 조작 불필요).

- [ ] **Step 4: 통과 확인**

Run: `venv/bin/pytest -q && venv/bin/flake8 src tests scripts`
Expected: 전체 PASS, flake8 클린.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/evaluate.py tests/test_cli.py
git commit -m "feat: evaluation script with aggregate statistics"
```

---

### Task 13: 학습 실행, 튜닝, 제출 모델 생성 (측정 기반 반복)

**Files:**
- Create: `models/1sess.txt`, `models/10sess.txt`, `models/100sess.txt`, `models/1000sess.txt`, `models/10000sess.txt`, `models/best.txt`, `docs/training-log.md`
- Modify (필요시): `src/config.py`

목표 기준 (수용 조건):
- **Mandatory**: best 모델이 100게임 평가에서 `max_length ≥ 10`,
  `p10_plus_rate ≥ 0.4` 권장.
- **Bonus 단계**: max_length 15 → 20 → 25 → 30 → 35 순서로 도전, 달성치 기록.

- [ ] **Step 1: 베이스라인 학습 및 평가**

```bash
make train
venv/bin/python -m scripts.evaluate models/100sess.txt -games 100
venv/bin/python -m scripts.evaluate models/10000sess.txt -games 200
```

Expected: 1 → 10 → 100 → 1000 → 10000 세션으로 갈수록 평가 길이가
증가(학습 진행 증명). 결과를 `docs/training-log.md`에 기록:

```markdown
# Training Log

## run 1 — baseline (binary12, G+20/R-20/M-1/D-100, a0.1 g0.95 e-decay0.995)
- cmd: make train (seed 42)
- eval 100sess: mean=?, max=?, p10+=?
- eval 10000sess: mean=?, max=?, p10+=?
```

- [ ] **Step 2: 미달 시 실험 매트릭스 (한 번에 한 변수만 변경, 각각 기록)**

우선순위 순:
1. 세션 수 증가: `-sessions 50000` (학습은 visual off + quiet로 수 분 내).
2. encoder 교체: `-encoder binary16` (self-trap 회피 개선 기대).
3. `config.py` reward 튜닝: `REWARD_MOVE`를 −0.5/−2.0으로,
   `REWARD_GREEN` +25, `REWARD_DEATH` −150 등.
4. `GAMMA` 0.99 (더 먼 보상 고려), `EPSILON_DECAY` 0.999(탐험 연장).

각 실험은 다음 패턴으로 실행·기록한다:
```bash
venv/bin/python snake -visual off -quiet -seed 42 -sessions 50000 \
    -encoder binary16 -save models/exp-b16-50k.txt
venv/bin/python -m scripts.evaluate models/exp-b16-50k.txt -games 200
```

- [ ] **Step 3: 최적 설정 확정 후 제출 모델 일괄 재생성**

최적 설정을 `src/config.py` 기본값으로 반영한 뒤:
```bash
make train
cp models/10000sess.txt models/best.txt   # 또는 최고 성능 모델
venv/bin/python -m scripts.evaluate models/best.txt -games 200
```

Expected: 수용 조건 충족. `docs/training-log.md`에 최종 설정·수치 기록.

- [ ] **Step 4: 디펜스용 진행도 확인 (1 vs 10 vs 100)**

```bash
venv/bin/python -m scripts.evaluate models/1sess.txt -games 50
venv/bin/python -m scripts.evaluate models/10sess.txt -games 50
venv/bin/python -m scripts.evaluate models/100sess.txt -games 50
```

Expected: 단조 증가 경향(학습 효과 시연 가능). 수치 기록.

- [ ] **Step 5: Commit**

```bash
git add models/ docs/training-log.md src/config.py
git commit -m "feat: trained models (1/10/100/1k/10k sessions) and log"
```

---

### Task 14: Bonus — board-size 일반화 검증

**Files:**
- Test: `tests/test_session.py` (추가)

- [ ] **Step 1: 실패하는(또는 검증) 테스트 추가**

`tests/test_session.py`에 추가:
```python
def test_same_model_runs_on_any_board_size(tmp_path):
    rng = random.Random(0)
    env = Environment(size=10, rng=rng)
    agent = QLearningAgent(rng=rng)
    SessionRunner(env, agent, verbose=False,
                  log_sessions=False).run(30)
    path = str(tmp_path / "model.txt")
    agent.save(path)
    for size in (5, 12, 20):
        rng2 = random.Random(1)
        loaded = QLearningAgent.load(path, rng=rng2)
        env2 = Environment(size=size, rng=rng2)
        runner = SessionRunner(env2, loaded, learning=False,
                               verbose=False, log_sessions=False)
        results = runner.run(3)
        assert len(results) == 3
```

- [ ] **Step 2: 테스트 실행**

Run: `venv/bin/pytest tests/test_session.py -v`
Expected: PASS (state encoder가 board-size 무관이므로 추가 구현 없이
통과해야 정상. 실패하면 encoder/Environment에 크기 종속이 섞인 것 —
원인 수정).

- [ ] **Step 3: 실모델 교차 평가 + 수동 데모 확인**

```bash
venv/bin/python -m scripts.evaluate models/best.txt -games 50
venv/bin/python -m scripts.evaluate models/best.txt -games 50 -board-size 15
venv/bin/python -m scripts.evaluate models/best.txt -games 50 -board-size 20
venv/bin/python snake -visual on -load models/best.txt -sessions 3 \
    -dontlearn -board-size 15
```

Expected: 모든 크기에서 정상 동작·합리적 성능(수치를
`docs/training-log.md`에 기록). GUI 창이 board 크기에 맞게 커짐.

- [ ] **Step 4: Commit**

```bash
git add tests/test_session.py docs/training-log.md
git commit -m "test: same trained model works across board sizes (bonus)"
```

---

### Task 15: Bonus — display 고도화 (lobby / 결과 화면 / 통계 패널)

**Files:**
- Modify: `src/display.py`, `src/session.py`, `src/cli.py`

- [ ] **Step 1: Display에 lobby와 결과 화면 추가**

`src/display.py`의 `Display`에 추가:
```python
    def show_lobby(self, lines):
        """Start screen: config summary; SPACE to start, Q to quit."""
        self.screen.fill(BG)
        title = pygame.font.SysFont(None, 42).render(
            "Learn2Slither", True, HEAD)
        self.screen.blit(title, (20, 20))
        for i, line in enumerate(lines):
            label = self.font.render(line, True, TEXT)
            self.screen.blit(label, (20, 80 + i * 28))
        hint = self.font.render(
            "SPACE: start    Q: quit", True, GREEN)
        self.screen.blit(hint, (20, 80 + len(lines) * 28 + 20))
        pygame.display.flip()
        return self._wait_for_step()

    def show_summary(self, lines):
        """End screen: session results; any key to close."""
        self.screen.fill(BG)
        title = pygame.font.SysFont(None, 42).render(
            "Results", True, GREEN)
        self.screen.blit(title, (20, 20))
        for i, line in enumerate(lines):
            label = self.font.render(line, True, TEXT)
            self.screen.blit(label, (20, 80 + i * 28))
        pygame.display.flip()
        while True:
            event = pygame.event.wait()
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return
```

- [ ] **Step 2: session/cli 연결**

`src/session.py`의 `run()` 마지막(`return results` 직전)에 추가:
```python
        if self.display is not None and results:
            lengths = [r[0] for r in results]
            durations = [r[1] for r in results]
            self.display.show_summary([
                f"sessions: {len(results)}",
                f"max length: {max_len}",
                f"max duration: {max_dur}",
                f"mean length: {sum(lengths) / len(lengths):.2f}",
                f"mean duration: "
                f"{sum(durations) / len(durations):.2f}",
            ])
```

`src/cli.py`의 `_run()`에서 `runner.run(args.sessions)` 직전에 추가:
```python
        if display is not None:
            started = display.show_lobby([
                f"sessions: {args.sessions}",
                f"board: {args.board_size}x{args.board_size}",
                f"model: {args.load or 'fresh agent'}",
                f"mode: {'evaluate' if args.dontlearn else 'train'}",
                f"speed: {args.speed} ms   "
                f"(+/- speed, P pause, SPACE step)",
            ])
            if not started:
                return
```

- [ ] **Step 3: lint + 회귀 + 수동 검증**

```bash
venv/bin/flake8 src tests scripts && venv/bin/pytest -q
venv/bin/python snake -visual on -load models/best.txt -sessions 3 -dontlearn
```
- [ ] lobby가 뜨고 SPACE로 시작, Q로 종료된다
- [ ] 종료 후 Results 화면에 max/mean 길이·duration이 표시된다
- [ ] visual off 경로는 기존과 동일하게 동작한다(테스트 PASS)

- [ ] **Step 4: Commit**

```bash
git add src/display.py src/session.py src/cli.py
git commit -m "feat: lobby, results screen and stats panel (bonus)"
```

---

### Task 16: README + 최종 검증 (디펜스 리허설)

**Files:**
- Create: `README.md`

- [ ] **Step 1: README 작성**

```markdown
# Learn2Slither

Q-learning으로 보드 환경을 학습하는 snake agent (42 RL project).

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Usage

    ./snake -sessions 10 -save models/10sess.txt -visual off
    ./snake -visual on -load models/100sess.txt -sessions 10 \
        -dontlearn -step-by-step
    ./snake -visual on -load models/best.txt

| Flag | Description |
|---|---|
| `-sessions N` | number of sessions (default 1) |
| `-save / -load PATH` | export / import the learning state |
| `-visual on/off` | pygame window on/off |
| `-dontlearn` | evaluation: no exploration, no Q update |
| `-step-by-step` | SPACE advances one step |
| `-speed MS` | display delay per step |
| `-board-size N` | bonus: any board size with the same model |
| `-seed N` | reproducibility |
| `-quiet` | hide per-step terminal vision/action |

Keys (visual): SPACE step/start, P pause, +/- speed, Q/ESC quit.

## Architecture

Environment(board rules) -> Interpreter(vision -> state, rewards)
-> Agent(Q-table, epsilon-greedy) -> Action -> Environment.
The agent only ever receives the state derived from the 4-direction
snake vision (subject IV.2).

## Models

`models/` contains states trained for 1, 10, 100, 1000, 10000
sessions plus `best.txt`. Regenerate with `make train`. Evaluate with
`python -m scripts.evaluate models/best.txt -games 100`.

## Quality

    make lint   # flake8 (42 norm)
    make test   # pytest
```

- [ ] **Step 2: 품질 게이트 전체 실행**

```bash
venv/bin/flake8 src tests scripts
venv/bin/pytest -q
```
Expected: flake8 0건, 테스트 전체 PASS.

- [ ] **Step 3: 디펜스 리허설 (빈 디렉토리 클론 시뮬레이션)**

```bash
TMP=$(mktemp -d) && git -C /Users/keokim/42/Learn2Slither archive HEAD \
    | tar -x -C "$TMP" && cd "$TMP"
python3 -m venv venv && venv/bin/pip install -r requirements.txt -q
venv/bin/python snake -sessions 10 -save /tmp/10sess.txt -visual off -quiet
venv/bin/python snake -visual off -load models/100sess.txt -sessions 2 \
    -dontlearn -quiet
venv/bin/flake8 src tests scripts && venv/bin/pytest -q
```

체크리스트 (모두 확인):
- [ ] 서브젝트 명령 3종이 모두 동작 (`-sessions/-save`, `-load/-dontlearn/-step-by-step`, `-load`만)
- [ ] `models/`에 1/10/100 세션 모델 존재 + 진행도 시연 가능
- [ ] best 모델 시연에서 길이 10+ 확인 (`-visual on -load models/best.txt`)
- [ ] vision 외 정보가 agent에 가지 않음(코드 리뷰: `choose_action(state)`만)
- [ ] board-size 변경 데모 동작 (`-board-size 15` + 동일 모델)
- [ ] 비정상 입력(없는 파일, 음수 세션)에서 메시지 후 정상 종료
- [ ] flake8 0건, pytest 전체 PASS

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, usage and architecture"
```

---

## Self-Review 결과

- **Spec coverage**: 설계 문서의 모든 요구사항이 Task 1–16에 매핑됨
  (보드 규칙→3·4, vision/state/reward→5·6, Q-learning→7·8, 루프→9,
  CLI→10, GUI→11, 모델 제출→13, bonus 3종→13·14·15, 품질/디펜스→16).
- **Placeholder 없음**: 모든 코드 블록은 완전한 구현이며 Step 단위로
  실행 명령과 기대 결과를 명시함. Task 13만 측정 기반 반복(실험 매트릭스와
  수용 조건 명시)으로 설계.
- **Type consistency**: `SessionRunner(env, agent, learning, step_by_step,
  verbose, display, log_sessions)`, `Display(board_size, speed_ms)`,
  `run(sessions) -> list[(max_length, duration)]`,
  `QLearningAgent.load(path, rng)` 시그니처가 전 Task에서 동일하게 사용됨.
