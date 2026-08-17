import json
import urllib.error
import urllib.request
from typing import Any


def _get(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestHealthAndState:
    def test_health(self, live_server):
        base_url, _ = live_server
        status, body = _get(f"{base_url}/health")
        assert status == 200
        assert body == {"status": "ok"}

    def test_state_shape(self, live_server):
        base_url, _ = live_server
        status, body = _get(f"{base_url}/state")
        assert status == 200
        assert body["current_player"] == 1
        assert body["winner"] is None
        assert body["board"]["size"] == 5


class TestValidEndpoints:
    def test_valid_move_true(self, live_server):
        base_url, _ = live_server
        status, body = _get(f"{base_url}/valid-move?player=1&direction=down")
        assert status == 200
        assert body["valid"] is True

    def test_valid_move_false_off_board(self, live_server):
        base_url, _ = live_server
        status, body = _get(f"{base_url}/valid-move?player=1&direction=up")
        assert status == 200
        assert body["valid"] is False

    def test_valid_move_bad_direction_returns_400(self, live_server):
        base_url, _ = live_server
        status, _ = _get(f"{base_url}/valid-move?player=1&direction=sideways")
        assert status == 400

    def test_valid_wall_true(self, live_server):
        base_url, _ = live_server
        status, body = _get(f"{base_url}/valid-wall?player=1&orientation=horizontal&row=0&col=0")
        assert status == 200
        assert body["valid"] is True


class TestMoveEndpoint:
    def test_move_success_updates_state(self, live_server):
        base_url, engine = live_server
        status, body = _post(f"{base_url}/move", {"player": 1, "direction": "down"})
        assert status == 200
        assert body["current_player"] == 2
        assert engine.board.p1_pos == [1, 2]

    def test_move_out_of_turn_returns_409(self, live_server):
        base_url, _ = live_server
        status, body = _post(f"{base_url}/move", {"player": 2, "direction": "up"})
        assert status == 409
        assert "error" in body

    def test_move_missing_field_returns_400(self, live_server):
        base_url, _ = live_server
        status, _ = _post(f"{base_url}/move", {"player": 1})
        assert status == 400

    def test_move_bad_direction_returns_400(self, live_server):
        base_url, _ = live_server
        status, _ = _post(f"{base_url}/move", {"player": 1, "direction": "sideways"})
        assert status == 400


class TestWallEndpoint:
    def test_wall_success(self, live_server):
        base_url, engine = live_server
        status, _ = _post(
            f"{base_url}/wall", {"player": 1, "orientation": "horizontal", "row": 0, "col": 0}
        )
        assert status == 200
        assert (0, 0) in engine.board.h_walls

    def test_wall_illegal_overlap_returns_409(self, live_server):
        base_url, _ = live_server
        status, _ = _post(
            f"{base_url}/wall", {"player": 1, "orientation": "horizontal", "row": 0, "col": 0}
        )
        assert status == 200
        status, body = _post(
            f"{base_url}/wall", {"player": 2, "orientation": "horizontal", "row": 0, "col": 1}
        )
        assert status == 409
        assert "error" in body


class TestClaimEndpoint:
    def test_first_claim_succeeds(self, live_server):
        base_url, _ = live_server
        status, body = _post(f"{base_url}/claim", {"player": 1})
        assert status == 200
        assert body == {"player": 1}

    def test_second_claim_for_same_player_is_rejected(self, live_server):
        base_url, _ = live_server
        status, _ = _post(f"{base_url}/claim", {"player": 1})
        assert status == 200

        status, body = _post(f"{base_url}/claim", {"player": 1})
        assert status == 409
        assert "error" in body

    def test_different_players_can_both_claim(self, live_server):
        base_url, _ = live_server
        status, _ = _post(f"{base_url}/claim", {"player": 1})
        assert status == 200
        status, _ = _post(f"{base_url}/claim", {"player": 2})
        assert status == 200

    def test_invalid_player_number_returns_400(self, live_server):
        base_url, _ = live_server
        status, _ = _post(f"{base_url}/claim", {"player": 3})
        assert status == 400
