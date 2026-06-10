"""Snake board environment: subject rules, spawning, step logic."""
import random
from enum import IntEnum

from src import config


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
