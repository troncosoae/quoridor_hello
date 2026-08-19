import argparse

from quoridor.agents import build_agent
from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.rendering import CLIRenderer
from quoridor.runner import GameRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a local Quoridor game.")
    parser.add_argument("--size", type=int, default=9, choices=[5, 7, 9])
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    parser.add_argument(
        "--agents", nargs="+", choices=["human", "bfs", "cnn", "mcts"],
        default=["human", "bfs"],
        help="One human/bfs/cnn/mcts token per player, e.g. --agents human bfs bfs bfs",
    )
    args = parser.parse_args()

    if len(args.agents) != args.players:
        parser.error(f"--players {args.players} requires exactly {args.players} --agents values")

    engine = QuoridorEngine(QuoridorBoard(args.size, args.players))
    agents = {
        player: build_agent(player, kind, args.players, args.size)
        for player, kind in enumerate(args.agents, start=1)
    }

    runner = GameRunner(engine, agents)
    winner = runner.run()

    print(CLIRenderer().render(engine.get_state()))
    print(f"Player {winner} wins!")


if __name__ == "__main__":
    main()
