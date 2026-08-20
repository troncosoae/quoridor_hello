from quoridor.actions import apply_action, legal_actions
from quoridor.agents import Agent
from quoridor.engine import EngineLike
from quoridor.game_store import GamePly, GameRecord, GameStore


class GameRunner:
    """Runs one game to completion. Recording is optional — pass a
    GameStore to have every ply (and the final outcome) saved there,
    tagged with each agent's KIND, on top of the exact same game loop.
    store=None (the default) behaves identically to before this capability
    existed — no game-execution code needs to know or care about training,
    storage, or anything else downstream of a recorded game."""

    def __init__(
        self,
        engine: EngineLike,
        agents: dict[int, Agent],
        store: GameStore | None = None,
        game_id: str = "",
        batch_index: int = 0,
        max_plies: int | None = None,
    ):
        self.engine = engine
        self.agents = agents
        self.store = store
        self.game_id = game_id
        self.batch_index = batch_index
        self.max_plies = max_plies

    def run(self) -> int | None:
        plies: list[GamePly] = []

        while True:
            winner = self.engine.winner()
            if winner is not None:
                break
            if self.max_plies is not None and len(plies) >= self.max_plies:
                break

            player = self.engine.current_player
            agent = self.agents[player]

            if self.store is not None:
                state = self.engine.get_state()
                legal = legal_actions(self.engine, player, state)
                action = agent.choose_action(self.engine)
                plies.append(GamePly(
                    state=state, current_player=player, action=action,
                    actor=agent.KIND, legal_actions=legal,
                ))
            else:
                action = agent.choose_action(self.engine)

            apply_action(self.engine, action)

        if self.store is not None:
            final_state = self.engine.get_state()
            self.store.save_game(GameRecord(
                game_id=self.game_id, batch_index=self.batch_index,
                size=final_state["size"], player_count=final_state["player_count"],
                plies=plies, winner=winner,
            ))

        return winner
