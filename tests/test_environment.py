import random

from src.environment import Action, DELTAS, Environment, Event


def make_env(seed=42, size=10):
    return Environment(size=size, rng=random.Random(seed))


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
