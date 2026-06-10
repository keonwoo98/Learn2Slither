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
