"""Untrained forward pass through QuoridorNet on a fresh board, wired
through the real engine/encoding so the shapes are verified end to end.
Weights are random — this is a plumbing check, not a policy demo."""

from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.rl.encoding import (
    action_size,
    encode_state,
    index_to_action,
    legal_action_mask,
    num_planes,
)
from quoridor.rl.network import QuoridorNet


def main() -> None:
    engine = QuoridorEngine(QuoridorBoard(size=9, player_count=2))
    state = engine.get_state()

    x = encode_state(state, engine.current_player).unsqueeze(0)
    mask = legal_action_mask(engine, engine.current_player, state["size"]).unsqueeze(0)

    net = QuoridorNet(
        size=state["size"],
        player_count=state["player_count"],
        in_channels=num_planes(state["player_count"]),
        action_size=action_size(state["size"]),
    )
    net.eval()

    policy, win_probs = net.predict(x, mask)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"policy shape: {tuple(policy.shape)}  (sums to {policy.sum().item():.3f})")
    print(f"win probs:    {win_probs.squeeze(0).tolist()}")

    best_index = int(policy.argmax(dim=-1).item())
    print(f"argmax action: {index_to_action(best_index, state['size'])}")


if __name__ == "__main__":
    main()
