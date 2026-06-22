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
    # board is square with a panel on the right, so width > height
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


def test_game_over_advances_on_space_in_step_mode():
    display = make_display()
    env = Environment(rng=random.Random(0))
    press(pygame.K_SPACE)
    assert display.show_game_over(env, "hit a wall",
                                  step_by_step=True) is True
    display.close()


def test_game_over_quit_returns_false():
    display = make_display()
    env = Environment(rng=random.Random(0))
    press(pygame.K_q)
    assert display.show_game_over(env, "hit a wall",
                                  step_by_step=True) is False
    display.close()


def test_game_over_auto_mode_can_be_quit():
    display = make_display()
    env = Environment(rng=random.Random(0))
    press(pygame.K_q)
    assert display.show_game_over(env, "hit a wall",
                                  step_by_step=False) is False
    display.close()


def test_space_pauses_in_auto_mode():
    display = make_display()
    assert display.paused is False
    press(pygame.K_SPACE)
    display.tick(step_by_step=False)
    assert display.paused is True
    display.close()


def test_p_resumes_from_pause():
    display = make_display()
    display.paused = True
    press(pygame.K_p)
    assert display.tick(step_by_step=False) is True
    assert display.paused is False
    display.close()
