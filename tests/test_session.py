import random

from src.agent import QLearningAgent
from src.environment import Action, Environment
from src.session import SessionRunner


def test_training_run_returns_one_result_per_session(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(rng=rng)
    runner = SessionRunner(env, agent, verbose=False, log_sessions=False)
    results = runner.run(5)
    assert len(results) == 5
    for length, duration in results:
        assert length >= 0 and duration > 0
    assert agent.trained_sessions == 5
    assert agent.q_table  # 학습이 일어나 Q-table이 채워짐
    out = capsys.readouterr().out
    assert "Game over, max length =" in out
    assert "max duration =" in out


def test_dontlearn_does_not_touch_agent(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(epsilon=0.5, rng=rng)
    runner = SessionRunner(env, agent, learning=False,
                           verbose=False, log_sessions=False)
    runner.run(3)
    assert agent.trained_sessions == 0
    assert agent.epsilon == 0.5
    assert all(v == [0.0] * 4 for v in agent.q_table.values())


def test_verbose_prints_vision_and_action(capsys):
    rng = random.Random(42)
    env = Environment(rng=rng)
    agent = QLearningAgent(rng=rng)
    runner = SessionRunner(env, agent, verbose=True, log_sessions=False)
    runner.run(1)
    out = capsys.readouterr().out
    assert "H" in out and "W" in out  # vision 출력
    assert any(a.name in out for a in Action)  # action 출력


class _CycleAgent:
    """2x2 사각형을 영원히 도는 스텁 — starvation cap 검증용."""

    encoder_name = "binary12"
    epsilon = 0.0

    def __init__(self):
        self._cycle = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        self._i = 0

    def choose_action(self, state, learning=True):
        action = self._cycle[self._i % 4]
        self._i += 1
        return action

    def update(self, *args):
        pass

    def end_session(self):
        pass


class _LoopEnv(Environment):
    """reset이 항상 같은 배치를 만드는 테스트용 환경."""

    def reset(self):
        super().reset()
        self.snake = [(0, 0)]
        self.green_apples = {(9, 9), (9, 8)}
        self.red_apples = {(8, 9)}
        self.alive = True
        self.duration = 0
        self.max_length = 1


def test_starvation_cap_ends_looping_session(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "MAX_STEPS_WITHOUT_FOOD", 20)
    env = _LoopEnv(rng=random.Random(0))
    runner = SessionRunner(env, _CycleAgent(), learning=False,
                           verbose=False, log_sessions=False)
    results = runner.run(1)
    assert results[0][1] == 20  # 20 스텝에서 강제 종료


def test_same_model_runs_on_any_board_size(tmp_path):
    rng = random.Random(0)
    env = Environment(size=10, rng=rng)
    agent = QLearningAgent(rng=rng)
    SessionRunner(env, agent, verbose=False,
                  log_sessions=False).run(30)
    path = str(tmp_path / "model.txt")
    agent.save(path)
    for size in (5, 12, 20):
        rng2 = random.Random(1)
        loaded = QLearningAgent.load(path, rng=rng2)
        env2 = Environment(size=size, rng=rng2)
        runner = SessionRunner(env2, loaded, learning=False,
                               verbose=False, log_sessions=False)
        results = runner.run(3)
        assert len(results) == 3
