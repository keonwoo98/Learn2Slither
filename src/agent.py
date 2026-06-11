"""Q-learning agent: sparse Q-table with an epsilon-greedy policy."""
import json
import random

from src import config
from src.environment import Action


class QLearningAgent:
    """Tabular Q-learning. The only input it ever gets is the state."""

    FORMAT_VERSION = 1

    def __init__(self, alpha=config.ALPHA, gamma=config.GAMMA,
                 epsilon=config.EPSILON_START,
                 encoder_name="binary16", rng=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.encoder_name = encoder_name
        self.rng = rng if rng is not None else random.Random()
        self.q_table = {}
        self.trained_sessions = 0

    def q_values(self, state):
        return self.q_table.setdefault(state, [0.0] * len(Action))

    def choose_action(self, state, learning=True):
        """Epsilon-greedy when learning; pure greedy otherwise."""
        if learning and self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))
        values = self.q_values(state)
        best = max(values)
        ties = [a for a in Action if values[a] == best]
        return self.rng.choice(ties)

    def update(self, state, action, reward, next_state, done):
        """One Q-learning backup for the (s, a, r, s') transition."""
        target = reward
        if not done:
            target += self.gamma * max(self.q_values(next_state))
        values = self.q_values(state)
        values[action] += self.alpha * (target - values[action])

    def end_session(self):
        """Decay exploration once per finished training session."""
        self.trained_sessions += 1
        self.epsilon = max(config.EPSILON_MIN,
                           self.epsilon * config.EPSILON_DECAY)

    def save(self, path):
        """Export the full learning state to a human-readable file."""
        data = {
            "format_version": self.FORMAT_VERSION,
            "state_encoder": self.encoder_name,
            "hyperparams": {"alpha": self.alpha, "gamma": self.gamma},
            "epsilon": self.epsilon,
            "trained_sessions": self.trained_sessions,
            "q_table": {str(s): v for s, v in self.q_table.items()},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    @classmethod
    def load(cls, path, rng=None):
        """Import a learning state previously written by save()."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not a valid model file: {path}") from exc
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {path}")
        try:
            agent = cls(
                alpha=data["hyperparams"]["alpha"],
                gamma=data["hyperparams"]["gamma"],
                epsilon=data["epsilon"],
                encoder_name=data["state_encoder"],
                rng=rng,
            )
            agent.trained_sessions = data["trained_sessions"]
            agent.q_table = {
                int(state): [float(v) for v in values]
                for state, values in data["q_table"].items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid model file: {path} ({exc})") from exc
        for values in agent.q_table.values():
            if len(values) != len(Action):
                raise ValueError(
                    f"invalid model file: {path} "
                    f"(q_table rows must have {len(Action)} values)")
        return agent
