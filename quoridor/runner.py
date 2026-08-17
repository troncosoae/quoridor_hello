from quoridor.actions import apply_action
from quoridor.agents import Agent
from quoridor.engine import EngineLike


class GameRunner:
    def __init__(self, engine: EngineLike, agents: dict[int, Agent]):
        self.engine = engine
        self.agents = agents

    def run(self) -> int:
        while True:
            winner = self.engine.winner()
            if winner is not None:
                return winner
            agent = self.agents[self.engine.current_player]
            apply_action(self.engine, agent.choose_action(self.engine))
