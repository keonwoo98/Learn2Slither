import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from src.display import Display  # noqa: E402
from src.environment import Environment  # noqa: E402


def make_display():
    return Display(10, speed_ms=1)


def press(key):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


def test_draw_and_tick_runs():
    display = make_display()
    env = Environment(rng=random.Random(0))
    display.draw(env, [("session", 1), ("length", 3),
                       ("epsilon", "1.00")])
    assert display.tick(step_by_step=False) is True
    display.close()


def test_draw_accepts_no_info():
    display = make_display()
    env = Environment(rng=random.Random(0))
    display.draw(env)
    display.close()


def test_window_has_side_panel():
    display = Display(10, speed_ms=1)
    width, height = display.screen.get_size()
    # 보드는 정사각형이고 오른쪽에 패널이 붙으므로 가로가 더 길다
    assert width > height
    assert width >= display.board_px + 200
    display.close()


def test_small_board_still_has_usable_window():
    display = Display(8, speed_ms=1)
    width, height = display.screen.get_size()
    assert width >= 600 and height >= 400
    display.close()


def test_step_by_step_advances_on_space():
    display = make_display()
    press(pygame.K_SPACE)
    assert display.tick(step_by_step=True) is True
    display.close()


def test_quit_key_stops():
    display = make_display()
    press(pygame.K_q)
    assert display.tick(step_by_step=True) is False
    display.close()


def test_minus_key_slows_down():
    display = make_display()
    display.speed_ms = 100
    press(pygame.K_MINUS)
    display.tick(step_by_step=False)
    assert display.speed_ms == 125
    display.close()


def test_lobby_starts_on_space_and_quits_on_q():
    display = make_display()
    press(pygame.K_SPACE)
    assert display.show_lobby(["sessions: 1"]) is True
    press(pygame.K_q)
    assert display.show_lobby(["sessions: 1"]) is False
    display.close()


def test_summary_closes_on_any_key():
    display = make_display()
    press(pygame.K_RETURN)
    display.show_summary(["max length: 10"])
    display.close()
