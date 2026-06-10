"""Q-learning agent: sparse Q-table with an epsilon-greedy policy."""
import random

from src import config
from src.environment import Action


class QLearningAgent:
    """Tabular Q-learning. The only input it ever gets is the state."""

    FORMAT_VERSION = 1

    def __init__(self, alpha=config.ALPHA, gamma=config.GAMMA,
                 epsilon=config.EPSILON_START,
                 encoder_name="binary12", rng=None):
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
