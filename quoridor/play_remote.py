import argparse
import sys
import time

from quoridor.actions import apply_action
from quoridor.agents import Agent, BFSAgent, CLIAgent
from quoridor.client import RemoteEngine, SeatTakenError
from quoridor.engine import InvalidMoveError
from quoridor.rendering import CLIRenderer

POLL_INTERVAL = 1.0


def _build_agent(player: int, kind: str) -> Agent:
    if kind == "human":
        return CLIAgent(player, CLIRenderer())
    return BFSAgent(player, CLIRenderer())


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect to a running Quoridor server.")
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--player", type=int, choices=[1, 2, 3, 4], default=None,
        help="Seat to claim. Omit to auto-assign the next open seat.",
    )
    parser.add_argument("--agent", choices=["human", "bfs"], required=True)
    args = parser.parse_args()

    remote = RemoteEngine(args.url)
    try:
        player = remote.claim_player(args.player)
    except SeatTakenError as e:
        requested = args.player if args.player is not None else "(auto-assign)"
        print(f"Could not connect as player {requested}: {e}")
        sys.exit(1)

    agent = _build_agent(player, args.agent)

    print(f"Connected to {args.url} as player {player} ({args.agent}).")

    while remote.winner() is None:
        if remote.current_player != player:
            time.sleep(POLL_INTERVAL)
            continue
        try:
            apply_action(remote, agent.choose_action(remote))
        except InvalidMoveError as e:
            print(f"Move rejected: {e}")

    print(f"Game over. Winner: player {remote.winner()}")


if __name__ == "__main__":
    main()
