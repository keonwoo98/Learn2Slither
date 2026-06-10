"""Pygame board display: speed control, pause and step-by-step mode."""
import pygame

CELL = 40
MARGIN = 1
PANEL_H = 40
BG = (24, 24, 24)
GRID = (52, 52, 52)
GREEN = (40, 200, 80)
RED = (220, 60, 60)
BODY = (60, 110, 220)
HEAD = (130, 180, 255)
TEXT = (235, 235, 235)


class Display:
    """Dedicated window for the board (subject IV.3 info box)."""

    def __init__(self, board_size, speed_ms):
        pygame.init()
        side = board_size * CELL
        self.speed_ms = speed_ms
        self.paused = False
        self.screen = pygame.display.set_mode((side, side + PANEL_H))
        pygame.display.set_caption("Learn2Slither")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

    def draw(self, env, info=""):
        self.screen.fill(BG)
        for row in range(env.size):
            for col in range(env.size):
                self._cell(row, col, GRID)
        for row, col in env.green_apples:
            self._cell(row, col, GREEN)
        for row, col in env.red_apples:
            self._cell(row, col, RED)
        for i, (row, col) in enumerate(env.snake):
            self._cell(row, col, HEAD if i == 0 else BODY)
        label = self.font.render(info, True, TEXT)
        self.screen.blit(label, (8, env.size * CELL + 10))
        pygame.display.flip()

    def _cell(self, row, col, color):
        rect = pygame.Rect(col * CELL + MARGIN, row * CELL + MARGIN,
                           CELL - 2 * MARGIN, CELL - 2 * MARGIN)
        pygame.draw.rect(self.screen, color, rect)

    def tick(self, step_by_step):
        """Pace one step and process input. False means user quit."""
        if step_by_step or self.paused:
            return self._wait_for_step()
        deadline = pygame.time.get_ticks() + self.speed_ms
        while pygame.time.get_ticks() < deadline:
            if not self._pump():
                return False
            self.clock.tick(120)
        return True

    def _pump(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_ms = max(10, self.speed_ms - 25)
                elif event.key == pygame.K_MINUS:
                    self.speed_ms = min(1000, self.speed_ms + 25)
                elif event.key == pygame.K_p:
                    self.paused = True
        return True

    def _wait_for_step(self):
        """Wait for SPACE (one step) or P (resume). False on quit."""
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    return True
                if event.key == pygame.K_p:
                    self.paused = False
                    return True

    def show_lobby(self, lines):
        """Start screen: config summary; SPACE to start, Q to quit."""
        self.screen.fill(BG)
        title = pygame.font.SysFont(None, 42).render(
            "Learn2Slither", True, HEAD)
        self.screen.blit(title, (20, 20))
        for i, line in enumerate(lines):
            label = self.font.render(line, True, TEXT)
            self.screen.blit(label, (20, 80 + i * 28))
        hint = self.font.render(
            "SPACE: start    Q: quit", True, GREEN)
        self.screen.blit(hint, (20, 80 + len(lines) * 28 + 20))
        pygame.display.flip()
        return self._wait_for_step()

    def show_summary(self, lines):
        """End screen: session results; any key to close."""
        self.screen.fill(BG)
        title = pygame.font.SysFont(None, 42).render(
            "Results", True, GREEN)
        self.screen.blit(title, (20, 20))
        for i, line in enumerate(lines):
            label = self.font.render(line, True, TEXT)
            self.screen.blit(label, (20, 80 + i * 28))
        pygame.display.flip()
        while True:
            event = pygame.event.wait()
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return

    def close(self):
        pygame.quit()
