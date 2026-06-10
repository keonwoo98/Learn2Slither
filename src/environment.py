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
