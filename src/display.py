"""Pygame display: a board panel on the left, live stats on the right."""
import pygame

MARGIN = 2
PANEL_W = 240
MIN_WIN_H = 420
TARGET_BOARD_PX = 540

BG = (16, 18, 27)
BOARD_BG = (28, 30, 42)
GRID = (44, 47, 62)
PANEL_BG = (22, 24, 34)
GREEN = (76, 209, 112)
RED = (232, 86, 86)
BODY = (66, 122, 240)
BODY_ALT = (92, 146, 250)
HEAD = (150, 200, 255)
EYE = (20, 22, 30)
TEXT = (228, 230, 238)
MUTED = (138, 144, 166)
ACCENT = (130, 180, 255)

KEYS = ["SPACE  pause / step", "P  resume",
        "+ / -  speed", "Q  quit"]


class Display:
    """Dedicated window: board on the left, live stats on the right."""

    def __init__(self, board_size, speed_ms):
        pygame.init()
        self.size = board_size
        self.cell = max(24, min(64, TARGET_BOARD_PX // board_size))
        self.board_px = self.cell * board_size
        self.win_w = self.board_px + PANEL_W
        self.win_h = max(self.board_px, MIN_WIN_H)
        self.speed_ms = speed_ms
        self.paused = False
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Learn2Slither")
        # Held SPACE repeats: tap = one step, hold = continuous stepping.
        pygame.key.set_repeat(300, 60)
        self.font = pygame.font.SysFont(None, 26)
        self.font_small = pygame.font.SysFont(None, 22)
        self.font_big = pygame.font.SysFont(None, 40)
        self.clock = pygame.time.Clock()

    # ---------- per-step drawing ----------

    def draw(self, env, info_lines=None):
        self.screen.fill(BG)
        self._draw_board(env)
        self._draw_side_panel(info_lines or [])
        pygame.display.flip()

    def _draw_board(self, env):
        pygame.draw.rect(self.screen, BOARD_BG,
                         (0, 0, self.board_px, self.board_px))
        for i in range(self.size + 1):
            offset = i * self.cell
            pygame.draw.line(self.screen, GRID, (offset, 0),
                             (offset, self.board_px))
            pygame.draw.line(self.screen, GRID, (0, offset),
                             (self.board_px, offset))
        for row, col in env.green_apples:
            self._apple(row, col, GREEN)
        for row, col in env.red_apples:
            self._apple(row, col, RED)
        for i, (row, col) in enumerate(env.snake):
            color = HEAD if i == 0 else (BODY if i % 2 else BODY_ALT)
            self._segment(row, col, color)
        if env.snake:
            self._eyes(env)

    def _segment(self, row, col, color):
        pad = MARGIN + 1
        rect = pygame.Rect(col * self.cell + pad, row * self.cell + pad,
                           self.cell - 2 * pad, self.cell - 2 * pad)
        pygame.draw.rect(self.screen, color, rect,
                         border_radius=max(3, self.cell // 5))

    def _apple(self, row, col, color):
        center = (col * self.cell + self.cell // 2,
                  row * self.cell + self.cell // 2)
        pygame.draw.circle(self.screen, color, center,
                           self.cell // 2 - MARGIN - 2)

    def _eyes(self, env):
        row, col = env.snake[0]
        d_row, d_col = self._head_dir(env.snake)
        cx = col * self.cell + self.cell / 2
        cy = row * self.cell + self.cell / 2
        radius = max(2, self.cell // 9)
        for sign in (-1, 1):
            ex = cx + d_col * self.cell * 0.17 - d_row * sign * \
                self.cell * 0.22
            ey = cy + d_row * self.cell * 0.17 + d_col * sign * \
                self.cell * 0.22
            pygame.draw.circle(self.screen, EYE, (int(ex), int(ey)),
                               radius)

    @staticmethod
    def _head_dir(snake):
        if len(snake) >= 2:
            (head_row, head_col), (neck_row, neck_col) = snake[0], snake[1]
            delta = (head_row - neck_row, head_col - neck_col)
            if delta != (0, 0):
                return delta
        return (-1, 0)

    def _draw_side_panel(self, info_lines):
        x0 = self.board_px
        pad = 22
        pygame.draw.rect(self.screen, PANEL_BG,
                         (x0, 0, PANEL_W, self.win_h))
        self.screen.blit(
            self.font.render("Learn2Slither", True, ACCENT),
            (x0 + pad, 26))
        y = 80
        for label, value in info_lines:
            self.screen.blit(
                self.font_small.render(str(label), True, MUTED),
                (x0 + pad, y))
            value_img = self.font.render(str(value), True, TEXT)
            self.screen.blit(
                value_img,
                (self.win_w - pad - value_img.get_width(), y - 2))
            y += 38
        key_y = self.win_h - len(KEYS) * 24 - 18
        for key in KEYS:
            self.screen.blit(
                self.font_small.render(key, True, MUTED),
                (x0 + pad, key_y))
            key_y += 24

    # ---------- pacing / input ----------

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
                elif event.key in (pygame.K_p, pygame.K_SPACE):
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

    # ---------- full-screen lobby / results ----------

    def _full_screen(self, title, lines, hint, hint_color):
        self.screen.fill(BG)
        self.screen.blit(self.font_big.render(title, True, ACCENT),
                         (40, 40))
        y = 120
        for line in lines:
            self.screen.blit(self.font.render(line, True, TEXT),
                             (40, y))
            y += 36
        self.screen.blit(self.font.render(hint, True, hint_color),
                         (40, y + 18))
        pygame.display.flip()

    def show_lobby(self, lines):
        """Start screen: config summary; SPACE to start, Q to quit."""
        self._full_screen("Learn2Slither", lines,
                          "SPACE: start     Q: quit", GREEN)
        return self._wait_for_step()

    def show_summary(self, lines):
        """End screen: session results; any key to close."""
        self._full_screen("Results", lines,
                          "press any key to close", GREEN)
        while True:
            event = pygame.event.wait()
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return

    def close(self):
        pygame.quit()
