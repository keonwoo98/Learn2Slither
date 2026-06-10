import random

from src import config
from src.environment import Environment, Event
from src.interpreter import (ENCODERS, cell_char, encode_binary12,
                             reward_for, vision_lines)


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
