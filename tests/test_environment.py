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
    # 머리 (5,5), 몸이 오른쪽→위로 꺾여 (5,6),(4,6),(4,5)
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
