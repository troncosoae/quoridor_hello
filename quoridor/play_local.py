import argparse

from quoridor.agents import Agent, BFSAgent, CLIAgent
from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.rendering import CLIRenderer
from quoridor.runner import GameRunner


def _build_agent(player: int, kind: str) -> Agent:
    if kind == "human":
        return CLIAgent(player, CLIRenderer())
    return BFSAgent(player, CLIRenderer())


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a local Quoridor game.")
    parser.add_argument("--size", type=int, default=9, choices=[5, 7, 9])
    parser.add_argument("--p1", choices=["human", "bfs"], default="human")
    parser.add_argument("--p2", choices=["human", "bfs"], default="bfs")
    args = parser.parse_args()

    engine = QuoridorEngine(QuoridorBoard(args.size))
    agents = {1: _build_agent(1, args.p1), 2: _build_agent(2, args.p2)}

    runner = GameRunner(engine, agents)
    winner = runner.run()

    print(CLIRenderer().render(engine.get_state()))
    print(f"Player {winner} wins!")


if __name__ == "__main__":
    main()
