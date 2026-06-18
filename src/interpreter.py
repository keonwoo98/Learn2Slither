"""Interpreter: board -> terminal vision, state encoding and rewards."""

from src import config
from src.environment import DELTAS, Action, Event


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


_VISION_COLORS = {
    "W": "\033[90m",     # wall: grey
    "H": "\033[1;96m",   # head: bright cyan, bold
    "S": "\033[94m",     # body: blue
    "G": "\033[92m",     # green apple
    "R": "\033[91m",     # red apple
    "0": "\033[2;37m",   # empty: dim
}
_RESET = "\033[0m"


def colorize_vision(lines):
    """Wrap each glyph in ANSI colour codes (for human-readable TTYs).

    The characters are unchanged; only colour escapes are added, so
    stripping the escapes reproduces the subject's exact vision format.
    """
    colored = []
    for line in lines:
        buffer = ""
        for char in line:
            color = _VISION_COLORS.get(char)
            buffer += (color + char + _RESET) if color else char
        colored.append(buffer)
    return colored
