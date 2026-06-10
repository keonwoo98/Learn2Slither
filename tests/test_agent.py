import random

from src import config
from src.agent import QLearningAgent
from src.environment import Action


def make_agent(epsilon=0.0, seed=0):
    return QLearningAgent(epsilon=epsilon, rng=random.Random(seed))


def test_greedy_picks_best_action():
    agent = make_agent(epsilon=0.0)
    agent.q_table[7] = [1.0, 5.0, 2.0, 3.0]
    assert agent.choose_action(7) == Action.LEFT


def test_unknown_state_initialized_to_zero():
    agent = make_agent()
    assert agent.q_values(123) == [0.0, 0.0, 0.0, 0.0]


def test_exploration_returns_valid_action():
    agent = make_agent(epsilon=1.0)
    for _ in range(20):
        assert agent.choose_action(0) in list(Action)


def test_learning_false_is_pure_greedy_even_with_high_epsilon():
    agent = make_agent(epsilon=1.0)
    agent.q_table[7] = [0.0, 0.0, 9.0, 0.0]
    for _ in range(20):
        assert agent.choose_action(7, learning=False) == Action.DOWN


def test_update_terminal_moves_toward_reward():
    agent = make_agent()
    agent.update(5, Action.UP, 10.0, 0, done=True)
    # Q = 0 + alpha * (10 - 0) = 1.0
    assert agent.q_table[5][Action.UP] == 1.0


def test_update_bootstraps_next_state():
    agent = make_agent()
    agent.q_table[2] = [0.0, 4.0, 0.0, 0.0]
    agent.update(1, Action.RIGHT, 1.0, 2, done=False)
    # target = 1 + 0.95*4 = 4.8 ; Q = 0 + 0.1*4.8 = 0.48
    assert abs(agent.q_table[1][Action.RIGHT] - 0.48) < 1e-9


def test_epsilon_decays_to_floor():
    agent = make_agent(epsilon=config.EPSILON_MIN * 1.001)
    agent.end_session()
    agent.end_session()
    assert agent.epsilon == config.EPSILON_MIN
    assert agent.trained_sessions == 2
