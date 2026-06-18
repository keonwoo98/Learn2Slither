"""Training / evaluation loop orchestration."""

from src import config
from src.environment import Event
from src.interpreter import ENCODERS, reward_for, vision_lines


class SessionRunner:
    """Runs sessions: env <-> interpreter <-> agent, plus I/O pacing."""

    def __init__(self, env, agent, learning=True, step_by_step=False,
                 verbose=True, display=None, log_sessions=True):
        self.env = env
        self.agent = agent
        self.learning = learning
        self.step_by_step = step_by_step
        self.verbose = verbose
        self.display = display
        self.log_sessions = log_sessions

    def run(self, sessions):
        """Run all sessions. Returns a list of (max_length, duration)."""
        encode = ENCODERS[self.agent.encoder_name]
        results = []
        stopped = False
        for num in range(1, sessions + 1):
            self.env.reset()
            steps_since_food = 0
            truncated = False
            while self.env.alive:
                if not self._render(num):
                    stopped = True
                    break
                state = encode(self.env)
                if self.verbose:
                    print("\n".join(vision_lines(self.env)))
                action = self.agent.choose_action(state, self.learning)
                if self.verbose:
                    print(action.name)
                event = self.env.step(action)
                if self.learning:
                    done = not self.env.alive
                    next_state = state if done else encode(self.env)
                    self.agent.update(state, action, reward_for(event),
                                      next_state, done)
                if event in (Event.ATE_GREEN, Event.ATE_RED):
                    steps_since_food = 0
                else:
                    steps_since_food += 1
                if steps_since_food >= config.MAX_STEPS_WITHOUT_FOOD:
                    truncated = True
                    break
            results.append((self.env.max_length, self.env.duration))
            if self.learning:
                self.agent.end_session()
            if self.log_sessions:
                note = ""
                if truncated:
                    note = (f" (truncated: "
                            f"{config.MAX_STEPS_WITHOUT_FOOD} steps "
                            f"without food)")
                print(f"Session {num}/{sessions}: "
                      f"max length = {self.env.max_length}, "
                      f"duration = {self.env.duration}{note}")
            if stopped:
                break
        max_len = max((r[0] for r in results), default=0)
        max_dur = max((r[1] for r in results), default=0)
        print(f"Game over, max length = {max_len}, "
              f"max duration = {max_dur}")
        if self.display is not None and results and not stopped:
            lengths = [r[0] for r in results]
            durations = [r[1] for r in results]
            self.display.show_summary([
                f"sessions: {len(results)}",
                f"max length: {max_len}",
                f"max duration: {max_dur}",
                f"mean length: {sum(lengths) / len(lengths):.2f}",
                f"mean duration: "
                f"{sum(durations) / len(durations):.2f}",
            ])
        return results

    def _render(self, session_num):
        """Draw the board and pace the loop. False means user quit."""
        if self.display is None:
            return True
        info = [
            ("session", session_num),
            ("length", self.env.length),
            ("duration", self.env.duration),
            ("epsilon", f"{self.agent.epsilon:.2f}"),
        ]
        self.display.draw(self.env, info)
        return self.display.tick(self.step_by_step)
