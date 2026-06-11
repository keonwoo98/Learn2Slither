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
| `-speed MS` | display delay per step (default 150) |
| `-board-size N` | bonus: any board size with the same model |
| `-seed N` | reproducibility |
| `-quiet` | hide per-step terminal vision/action |
| `-encoder NAME` | state encoder for a fresh agent |

Keys (visual): SPACE start/step, P pause/resume, +/- speed, Q/ESC quit.

## Architecture

Environment(board rules) -> Interpreter(vision -> state, rewards)
-> Agent(Q-table, epsilon-greedy) -> Action -> Environment.

The agent only ever receives a single integer state derived from the
4-direction snake vision (subject IV.2) — never the full board. The
state encodes, per direction: danger adjacent, green apple seen, red
apple seen, body seen (relative features, so the same model plays on
any board size).

The board implements exactly the subject's three game-over rules
(wall, own tail, length 0). Independently of those, the session
runner truncates a session after 500 consecutive steps without
eating — an anti-livelock guard (a greedy policy can loop forever);
truncated sessions are marked as such in the per-session log.

## Models

`models/` contains learning states trained for 1, 10, 100, 1000,
10000 and 50000 sessions (`best.txt` = 50000), plus an alternate
update strategy (`alt-binary12-50k.txt`, reduced state encoder).
Performance of `best.txt` over 100 greedy games on 10x10: mean
length 31.6, median 35.5, max 51, 96% of games reach length 10+.
See `docs/training-log.md` for the full experiment matrix.

Regenerate everything with `make train`. Evaluate any model with:

    python -m scripts.evaluate models/best.txt -games 100

## Quality

    make lint   # flake8 (42 norm)
    make test   # pytest (51 tests)
