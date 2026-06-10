"""Command-line interface using the subject's single-dash flags."""
import argparse
import random
import sys

from src import config
from src.agent import QLearningAgent
from src.environment import Environment
from src.interpreter import ENCODERS
from src.session import SessionRunner


def build_parser():
    parser = argparse.ArgumentParser(
        prog="snake",
        description="Learn2Slither: a snake that learns by Q-learning")
    parser.add_argument("-sessions", type=int, default=1,
                        help="number of sessions to run")
    parser.add_argument("-save", metavar="PATH",
                        help="save the learning state when done")
    parser.add_argument("-load", metavar="PATH",
                        help="load a learning state before running")
    parser.add_argument("-visual", choices=["on", "off"], default="on",
                        help="show the pygame board window")
    parser.add_argument("-dontlearn", action="store_true",
                        help="evaluation mode: no exploration/updates")
    parser.add_argument("-step-by-step", dest="step_by_step",
                        action="store_true",
                        help="advance one step per SPACE key press")
    parser.add_argument("-speed", type=int,
                        default=config.DEFAULT_SPEED_MS,
                        help="display delay per step (ms)")
    parser.add_argument("-board-size", dest="board_size", type=int,
                        default=config.BOARD_SIZE,
                        help="board size (bonus)")
    parser.add_argument("-seed", type=int, help="random seed")
    parser.add_argument("-quiet", action="store_true",
                        help="suppress per-step terminal output")
    parser.add_argument("-encoder", choices=sorted(ENCODERS),
                        default="binary16",
                        help="state encoder for a fresh agent")
    return parser


def _run(args):
    rng = random.Random(args.seed)
    if args.load:
        agent = QLearningAgent.load(args.load, rng=rng)
        print(f"Load trained model from {args.load}")
    else:
        agent = QLearningAgent(encoder_name=args.encoder, rng=rng)
    if agent.encoder_name not in ENCODERS:
        raise ValueError(
            f"unknown state encoder '{agent.encoder_name}' in model")
    env = Environment(size=args.board_size, rng=rng)
    display = None
    try:
        if args.visual == "on":
            from src.display import Display
            display = Display(args.board_size, args.speed)
        runner = SessionRunner(
            env, agent,
            learning=not args.dontlearn,
            step_by_step=args.step_by_step,
            verbose=not args.quiet,
            display=display)
        runner.run(args.sessions)
    finally:
        if display is not None:
            display.close()
    if args.save:
        agent.save(args.save)
        print(f"Save learning state in {args.save}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sessions < 1:
        parser.error("-sessions must be >= 1")
    if args.board_size < 5:
        parser.error("-board-size must be >= 5")
    if args.speed < 1:
        parser.error("-speed must be >= 1")
    try:
        _run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # 서브젝트 규칙: 예기치 않은 크래시 금지
        print(f"snake: error: {exc}", file=sys.stderr)
        return 1
    return 0
