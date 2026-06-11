#!/usr/bin/env python3
"""Evaluate a trained model over many games and print statistics."""
import argparse
import random
import statistics
import sys

from src.agent import QLearningAgent
from src.environment import Environment
from src.session import SessionRunner


def evaluate(model_path, games=100, board_size=10, seed=42):
    """Run games in dontlearn mode; return aggregate statistics."""
    rng = random.Random(seed)
    agent = QLearningAgent.load(model_path, rng=rng)
    env = Environment(size=board_size, rng=rng)
    runner = SessionRunner(env, agent, learning=False,
                           verbose=False, log_sessions=False)
    results = runner.run(games)
    lengths = [r[0] for r in results]
    durations = [r[1] for r in results]
    return {
        "games": len(results),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "max_length": max(lengths),
        "p10_plus_rate": sum(x >= 10 for x in lengths) / len(lengths),
        "mean_duration": statistics.mean(durations),
        "max_duration": max(durations),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("-games", type=int, default=100)
    parser.add_argument("-board-size", dest="board_size", type=int,
                        default=10)
    parser.add_argument("-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("-games must be >= 1")
    if args.board_size < 5:
        parser.error("-board-size must be >= 5")
    try:
        stats = evaluate(args.model, args.games, args.board_size,
                         args.seed)
    except (OSError, ValueError) as exc:
        print(f"evaluate: error: {exc}", file=sys.stderr)
        return 1
    print(f"model: {args.model}  board: {args.board_size}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
