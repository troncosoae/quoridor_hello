from dataclasses import dataclass

from quoridor.actions import Action
from quoridor.board import BoardState
from quoridor.game_store import GameRecord
from quoridor.state_key import state_key


@dataclass
class StateTarget:
    state: BoardState
    """Needed to re-encode the network input at train time — state_key is
    a one-way hash, there's no reversing it back into a board."""
    current_player: int
    legal_actions: list[Action]
    """From the representative ply — needed to build a legal-action mask
    at train time even for states whose policy target ends up empty."""
    value: list[float]
    """Normalized win-count vector, length player_count."""
    policy: dict[Action, float]
    """Normalized one-step-lookahead policy target. May be empty for a
    state that was only ever visited via non-model (e.g. bfs) plies — see
    the actor filter below."""


def _normalize(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [1.0 / len(counts)] * len(counts)
    return [c / total for c in counts]


def _normalize_policy(weights: dict[Action, float], temperature: float) -> dict[Action, float]:
    sharpened = {action: weight ** (1.0 / temperature) for action, weight in weights.items()}
    total = sum(sharpened.values())
    if total <= 0.0:
        # Every recorded successor was a certain loss for the mover (e.g.
        # late in an already-lost game) — fall back to uniform over the
        # actions actually observed, rather than dividing by zero.
        uniform = 1.0 / len(sharpened)
        return dict.fromkeys(sharpened, uniform)
    return {action: weight / total for action, weight in sharpened.items()}


def derive_training_targets(
    games: list[GameRecord], policy_temperature: float = 1.0
) -> dict[str, StateTarget]:
    """Two-pass Monte Carlo value + one-step-lookahead policy derivation.

    Pass 1: every-visit Monte Carlo state value, from ALL plies regardless
    of actor — a real win/loss against a baseline agent is still valid
    grounding for "who tends to win from this state." Every-visit (not
    first-visit) is deliberate: a same-(state, current_player) repeat is
    reachable within one game (walls only ever accumulate, so an exact
    repeat just needs zero wall placements between visits plus both pawns
    returning to the same cells with the same player to move — nothing in
    the engine prevents this).

    Pass 2: policy target ONLY from plies where the actor was "model".
    TwoPlayerBFSAgent (and any other fixed baseline) never places walls —
    folding its deterministic plies into the policy target would teach the
    network "never place a wall here" purely because the baseline doesn't
    know how to, not because it's actually bad. Value derivation is
    unaffected by this filter — every ply's outcome still counts there.
    """
    win_counts: dict[str, list[int]] = {}
    representative: dict[str, tuple[BoardState, int, list[Action]]] = {}

    for game in games:
        if game.winner is None:
            continue
        for ply in game.plies:
            key = state_key(ply.state, ply.current_player)
            win_counts.setdefault(key, [0] * game.player_count)
            representative.setdefault(key, (ply.state, ply.current_player, ply.legal_actions))
            win_counts[key][game.winner - 1] += 1

    values = {key: _normalize(counts) for key, counts in win_counts.items()}

    action_weight_samples: dict[str, dict[Action, list[float]]] = {}
    for game in games:
        if game.winner is None:
            continue
        for i, ply in enumerate(game.plies):
            if ply.actor != "model":
                continue

            key = state_key(ply.state, ply.current_player)
            if i + 1 < len(game.plies):
                next_ply = game.plies[i + 1]
                next_key = state_key(next_ply.state, next_ply.current_player)
                # Pass 2 only ever iterates games Pass 1 already fully
                # covered, so next_key is definitionally already in
                # values. If this ever fires, Pass 1/Pass 2 have desynced
                # — fail loud rather than silently substituting a
                # misleading uniform value.
                assert next_key in values, f"successor state {next_key} missing from values"
                successor_value = values[next_key]
            else:
                # Last ply = the winning move itself. Its resulting
                # (terminal) state is never itself recorded as a ply (no
                # move is made FROM a won position) — but the outcome is
                # known exactly, no lookup needed.
                successor_value = [0.0] * game.player_count
                successor_value[game.winner - 1] = 1.0

            weight = successor_value[ply.current_player - 1]
            action_weight_samples.setdefault(key, {}).setdefault(ply.action, []).append(weight)

    policies: dict[str, dict[Action, float]] = {}
    for key, per_action in action_weight_samples.items():
        averaged = {action: sum(ws) / len(ws) for action, ws in per_action.items()}
        policies[key] = _normalize_policy(averaged, policy_temperature)

    return {
        key: StateTarget(
            state=representative[key][0],
            current_player=representative[key][1],
            legal_actions=representative[key][2],
            value=values[key],
            policy=policies.get(key, {}),
        )
        for key in values
    }
