from src.environment import Action, DELTAS, Event


def test_actions_are_up_left_down_right():
    assert [a.name for a in Action] == ["UP", "LEFT", "DOWN", "RIGHT"]


def test_deltas_move_one_cell():
    assert DELTAS[Action.UP] == (-1, 0)
    assert DELTAS[Action.LEFT] == (0, -1)
    assert DELTAS[Action.DOWN] == (1, 0)
    assert DELTAS[Action.RIGHT] == (0, 1)


def test_events_exist():
    assert {e.name for e in Event} == {
        "MOVED", "ATE_GREEN", "ATE_RED", "DIED"}
