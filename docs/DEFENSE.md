# Learn2Slither — Defense Script (English)

> Follows the Intra evaluation scale, item by item.
> Each step is split into **💬 Say it / ⌨️ Run / 📂 Show code**.
> Assumes the evaluator may not know the project, so it starts with context.

---

## 0. Intro — what this project is

💬
> "Hi! This project is **Learn2Slither** — it's about **reinforcement
> learning**. I build a snake that **learns to play the game by itself**.
>
> The key idea is that I never coded 'move like this'. It's like training a
> dog: I only give a **reward when it does well and a penalty when it does
> badly**, and the snake learns to maximize that reward on its own.
>
> The snake can only see in **4 directions from its head** — up, down, left,
> right — and it decides where to go based only on that vision. The goal is
> to eat apples, grow to a **length of at least 10**, and survive as long as
> possible.
>
> The learning algorithm is **Q-learning**: the snake keeps a kind of
> **score table (a Q-table)** that records, for each situation, how good
> each direction is.
>
> The structure follows the subject: three modules —
> **Environment / Interpreter / Agent** — and it's written in Python. Let me
> walk you through it in the order of the evaluation sheet."

---

## 1. Pre-check — clone & norm

💬
> "First, you clone my repo into an empty folder, make a virtualenv, and
> install the dependencies — then it just runs. The exact versions are
> pinned in `requirements.txt`."

⌨️ (skip if already cloned)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

💬
> "Since it's Python, we need the **norm check (flake8)**. Let me run it —
> no output means it passes."

⌨️
```bash
flake8 src tests scripts
```

💬
> "I also wrote tests that automatically check the rules, the learning, and
> save/load. Let me run them."

⌨️
```bash
python -m pytest -q
```
> "All green. Let's start the actual evaluation."

---

## 2. Part 1 — Environment / Board

💬
> "Part 1 is: does the board look like Snake, and are the rules correct? Let
> me **start a training session with a fresh model** so you can see. It's
> untrained, so it dies fast, but that's enough to check the board rules."

⌨️
```bash
./snake -visual on -sessions 5
```
> (press SPACE to start)

💬 (pointing at the window)
> "So you can see a **10×10 board**, with **2 green apples and 1 red apple**
> placed randomly. The snake starts at **length 3** in a random spot. Eating
> a green apple grows it by 1; hitting a wall or its own body kills it. On
> the right panel you can see the **current length and what killed it last
> (last death)**.
>
> The speed is set so a human can follow it — 150ms by default — and you can
> change it with `+` / `−`."

📂 if asked to show the rules in code:
> "All the rules live in `src/environment.py`."
- **`step()` (environment.py:83)** — "This is the core of one move. It
  computes the new head position (lines 87–88), and if it's **out of bounds
  it dies as 'hit a wall'**, or if it **overlaps the body, 'hit its own
  body'** (lines 89–96). Eating green keeps the tail, so +1; red removes the
  tail twice, so −1; an empty cell removes it once, so the length stays."
- **`_place_snake()` (environment.py:62)** — "Places the snake at a random
  spot, length 3 in a straight line, picking a direction that stays inside
  the board."
- **`_spawn()` (environment.py:74)** — "Apples only appear on cells that
  aren't the snake or another apple."
- **`config.py:3–6`** — "Values like 10×10, 2/1 apples, length 3 are
  collected here."

---

## 3. Part 2 — State (most important: the −42 trap)

💬
> "Part 2 is the snake's **vision (state)**. Two things to check: that the
> vision is printed to the terminal on every move, and — most importantly —
> that I **don't give the agent any information beyond that vision**. Giving
> more is a −42 penalty, so this is the critical one.
>
> Looking at the terminal, every step prints this cross shape."

⌨️ (point at the output already showing, or)
```bash
./snake -visual off -load models/best.txt -dontlearn -seed 5
```

💬
> "This cross is the snake's vision. **W is wall, H is head, S is body, G is
> green apple, R is red apple, 0 is empty.** It's what the head sees in the
> four directions, all the way to the wall. I also colorized it for
> readability. The next line shows the action the snake chose (`-> RIGHT`)."

📂 proving "only vision is passed" in code (this is the key part):
> "Let me prove in code that the agent only ever gets the vision. I'll follow
> the flow."
- **`session.py:36`** — "`state = encode(self.env)` — it compresses the
  board into a single number."
- **`session.py:39`** — "`agent.choose_action(state, ...)` — the agent
  receives **only this integer state**. Never the board object, coordinates,
  or total length."
- **`interpreter.py:_ray (37) / encode_binary16 (64)`** — "That state is
  built here: it only scans the **4 rays from the head**, and for each
  direction encodes four yes/no bits — 'is it dangerous / green seen / red
  seen / my body seen'. So it's derived **only from what's visible**."
- **`agent.py:28`** — "The agent's `choose_action` takes just `state`. It
  doesn't even import environment — only the Action enum."

💬
> "So the agent only gets what the snake can actually see. No rule
> violation."

---

## 4. Part 3 — Action

💬
> "Part 3 is the action. The agent can only pick **one of UP, LEFT, DOWN,
> RIGHT**, and the chosen one is printed in the terminal — like the
> `-> RIGHT` you saw. And the snake actually moves that way."

📂
- **`environment.py:8` (Action)** — "Actions are defined as exactly these
  four. There's no fifth one."
- **`environment.py:15` (DELTAS)** — "This maps each action to a coordinate
  change — UP is row −1, and so on."
- **`session.py:39–41`** — "Here it picks the action and prints it."

---

## 5. Part 4 — Rewards

💬
> "Part 4 is rewards. Let me **show the snake eating apples** and explain the
> scoring. With a trained model it eats well."

⌨️
```bash
./snake -visual on -load models/best.txt -dontlearn
```
> (snake eats green apples and grows)

💬
> "I designed the rewards like this:"
- "Green apple = **+20** (good)"
- "Red apple = **−20** (bad)"
- "Plain move = **−1** (don't waste time, head for the apple)"
- "Death = **−100** (the biggest penalty)"
>
> "The key is the **order of magnitude**. Death has to hurt the most so the
> snake learns to stay alive first, and the small penalty on moving pushes it
> toward apples. If death were cheap, it might learn to risk dying just to
> grab an apple."

📂
- **`config.py:8–11`** — "The reward values are here."
- **`interpreter.py:reward_for (85)`** — "This function turns what happened
  in the game (ate green, died, etc.) into the score the agent receives."

---

## 6. Part 5 — Q-learning

💬
> "Part 5 is the core, Q-learning. Let me explain **what a Q-value is and how
> it's computed.**
>
> A Q-value is basically **the expected 'how good is it' score for going in
> this direction from this situation**. The snake stores these in a
> **Q-table**. For each situation it has a score for the four directions, and
> it goes for the highest one.
>
> At first the table is blank (all zeros), so it wanders and dies. But when
> it dies it gets −100, and that lowers the score for that situation — it
> learns 'don't go here'. Repeat that tens of thousands of times and the
> table gets smart."

📂
- **`agent.py:update (37)`** — "This is the heart of learning. The formula
  is: `new Q = old Q + learning_rate × (reward + discount × best Q of the
  next state − old Q)`. The **learning rate 0.1** means each experience is
  applied slowly; the **discount 0.99** means it values future rewards
  almost as much as immediate ones — that's what makes the value of the
  'path toward an apple' rise. And **when it dies there's no future, so only
  the reward is used** (line 40, `if not done`)."
- **`agent.py:choose_action (28)`** — "Action selection. Usually it takes
  the highest-scoring direction (exploitation), but **sometimes it moves
  randomly (exploration)** — otherwise it would never discover better paths.
  That exploration probability (epsilon) is the 'random' the subject asks
  for, and it decreases as learning progresses (`end_session`, line 45)."
- "I used **only a Q-table** — no pathfinding or any other algorithm."

---

## 7. Features and structure

💬
> "This item checks several extra features at once. Let me go through them."

**(1) Train with a defined number of sessions**
⌨️
```bash
./snake -sessions 100 -visual off -save models/demo.txt
```
> "`-sessions` sets the number of training games, and turning the display off
> (`-visual off`) makes it train almost instantly."

**(2) Save/load + models at various training levels**
⌨️
```bash
ls models/
```
> "I just saved one with `-save`. The `models/` folder has models trained for
> **1, 10, 100, 1000, 10000, and 50000 sessions** — to show how it gets
> smarter as training adds up."

📂
- **`agent.py:save (51) / load (65)`** — "Models are saved as JSON text. It
  stores not just the Q-table but also epsilon and the session count, so I
  can resume training later. On load, a corrupted file is handled with a
  clean error."

**(3) Use without learning (for verification)**
> "With the `-dontlearn` flag it doesn't touch the table — it just tests the
> skill, so I can measure a model's real performance."
📂 **`session.py`, the `if self.learning:` line** — "This one line turns off
learning."

**(4) step-by-step + structure**
⌨️
```bash
./snake -visual on -load models/best.txt -dontlearn -step-by-step
```
> "With `-step-by-step`, it advances **one cell per SPACE press**, so I can
> inspect the vision and the action one step at a time. Hold SPACE for
> continuous stepping."
>
> "The structure follows the subject's diagram: **environment (the board) →
> interpreter (vision and reward) → agent (the brain)**, and session is the
> coordinator that calls them in order."

---

## 8. Testing trained models (the 50% rule)

💬
> "This item has a quantitative bar: **with the best model run without
> learning, does it reach length 10 more than 50% of the time?** Let me run
> my evaluation script over 100 games and show the stats."

⌨️
```bash
python -m scripts.evaluate models/best.txt -games 100
```
> (result: mean_length ~31, p10_plus_rate ~0.96 ...)

💬
> "As you can see, it reaches length 10 in about **96% of games** — well
> above the 50% bar. The average length is over 30. If you want to watch one
> game live, I can show it in the window."

⌨️ (optional)
```bash
./snake -visual on -load models/best.txt -dontlearn
```

---

## 9. Bonus — Length test (0–5 points)

💬
> "Now the bonuses. The first gives **points by achieved length** — one point
> per 5 units, full marks at 35 or more. As you just saw, the **median is
> around 35 and the max is over 50**. Over a few games you'll almost
> certainly see it pass 35."

⌨️ (check several games quickly)
```bash
python -m scripts.evaluate models/best.txt -games 50
```
> "You can look at the max and the distribution here."

---

## 10. Bonus Part 1 — visuals (start/end interfaces + score)

💬
> "The second bonus is whether the display is appealing and shows the score
> on start and end screens."

⌨️
```bash
./snake -visual on -load models/best.txt -dontlearn -sessions 3
```

💬
> "At the start there's a **lobby screen** showing the session count, board
> size, model, and mode — press SPACE to start. During the game the **right
> panel** shows the session number, current length, survival time, and the
> last death cause in real time. When it's done, a **results screen** shows
> stats like max and mean length."

📂 (if asked)
- **`display.py:show_lobby / show_summary`** — "Start and end screens."
- **`display.py:_draw_side_panel`** — "The live stats panel."

---

## 11. Bonus Part 2 — changing board size

💬
> "The last bonus: **does the same model file still play on a different board
> size?** The sheet asks for at least length 7 on sizes between 8×8 and
> 15×15. Let me run the same `best.txt` with just the size changed."

⌨️
```bash
python -m scripts.evaluate models/best.txt -games 30 -board-size 8
python -m scripts.evaluate models/best.txt -games 30 -board-size 15
```
> "Both 8×8 and 15×15 go well past length 7. I can show it in the window
> too."

⌨️ (optional)
```bash
./snake -visual on -load models/best.txt -dontlearn -board-size 15
```

💬
> "This works because my state encoding uses **relative information from the
> head** — 'is there an apple above me, yes or no' — not absolute
> coordinates. So when the board size changes, the same model still works."

📂 **`interpreter.py:encode_binary16 (64)`** — "Here it only checks
'present / not present', not distance, so it's size-independent."

---

## 12. Wrap-up — stability (no crashes)

💬
> "Finally, the program never crashes on bad input — it gives a clean error.
> For example, loading a model file that doesn't exist."

⌨️ (if the evaluator probes stability)
```bash
./snake -load /no-such-file.txt -visual off
./snake -sessions 0 -visual off
```
> "You see — no traceback, just a single `snake: error: ...` line."

📂 **`cli.py:main (85)`** — "The whole run is wrapped in try/except, so any
error becomes a clean message and exit code — to respect the subject's
'no unexpected crash' rule."

💬
> "That's the whole project. Happy to show any part in code — thanks!"

---

## Appendix A — likely follow-up questions

**Q. Why does the snake sometimes trap itself and die?**
> "Its vision is only 4 directions, so it can't see the whole layout. It
> can't perfectly avoid boxing itself in. But I added a 'is my body in this
> direction' bit to the binary16 encoding, which improved it a lot."

**Q. The 1/10/100-session models are barely length 3?**
> "Right. Early on there's a lot of exploration (random moves), so learning
> is slow. The subject's own example has a 10-session model at length ~4. It
> picks up noticeably from 1000 sessions, and 50000 is the best model. I can
> compare 1sess vs best with `evaluate` to show the learning curve."

**Q. What's that starvation / 500-step thing?**
> "It's a safeguard against an infinite loop where the snake circles forever
> without eating. It's not a board rule — it's a cutoff in the session loop,
> and it's shown as 'truncated' in the terminal."

**Q. Can you train a fresh model live?**
> "Sure." → `./snake -sessions 1000 -visual off -save models/live.txt` →
> `python -m scripts.evaluate models/live.txt -games 30`

---

## Appendix B — command cheat-sheet

```bash
# norm + tests
flake8 src tests scripts
python -m pytest -q

# Part 1–3: board / vision / action with a fresh (blank) model
./snake -visual on -sessions 5

# Part 4–6 + Testing + Bonus1: trained model demo
./snake -visual on -load models/best.txt -dontlearn

# step through one cell at a time (observe a death, etc.)
./snake -visual on -load models/best.txt -dontlearn -step-by-step

# 50% rule + Bonus Length: statistics
python -m scripts.evaluate models/best.txt -games 100

# Bonus2: same model on different board sizes
python -m scripts.evaluate models/best.txt -games 30 -board-size 8
python -m scripts.evaluate models/best.txt -games 30 -board-size 15

# train live (fast)
./snake -sessions 100 -visual off -save models/demo.txt

# crash safety
./snake -load /no-such-file.txt -visual off
```
