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
