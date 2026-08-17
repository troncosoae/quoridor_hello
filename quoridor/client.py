import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from quoridor.board import BoardState
from quoridor.engine import Direction, EngineLike, InvalidMoveError, WallOrientation


class SeatTakenError(Exception):
    """A player slot on the server is already claimed by another connection."""


class RemoteEngine:
    """EngineLike implementation that talks to a quoridor.server over HTTP.

    Talks JSON over plain urllib — no extra dependency. `current_player`
    is a plain cached attribute (never a property that does hidden network
    I/O on read); it's refreshed explicitly by get_state()/move()/place_wall().
    """

    def __init__(self, base_url: str, connect_retries: int = 10, retry_delay: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.current_player: int = 1
        self._winner: int | None = None
        self._wait_for_server(connect_retries, retry_delay)
        self._refresh()

    def _wait_for_server(self, retries: int, delay: float) -> None:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                self._get("/health")
                return
            except OSError as e:
                last_error = e
                time.sleep(delay)
        raise ConnectionError(f"Could not reach server at {self.base_url}") from last_error

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        with urllib.request.urlopen(f"{self.base_url}{path}{query}") as response:
            return dict(json.loads(response.read()))

    def _request(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, dict(json.loads(response.read()))
        except urllib.error.HTTPError as e:
            return e.code, dict(json.loads(e.read()))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self._request(path, payload)
        if status == 409:
            raise InvalidMoveError(body.get("error", "invalid move")) from None
        return body

    def claim_player(self, player: int) -> None:
        """Register this connection as the one controlling `player`.

        Must be called before acting for that player. Raises SeatTakenError
        if the server already has another connection claiming the same
        player number — this is what stops two clients from both playing
        as player 1.
        """
        status, body = self._request("/claim", {"player": player})
        if status == 409:
            raise SeatTakenError(
                body.get("error", f"player {player} is already connected")
            ) from None

    def _refresh(self) -> BoardState:
        data = self._get("/state")
        self.current_player = data["current_player"]
        self._winner = data["winner"]
        return dict(data["board"])  # type: ignore[return-value]

    def get_state(self) -> BoardState:
        return self._refresh()

    def winner(self) -> int | None:
        self._refresh()
        return self._winner

    def is_valid_move(self, player: int, direction: Direction) -> bool:
        result = self._get("/valid-move", {"player": str(player), "direction": direction.value})
        return bool(result["valid"])

    def move(self, player: int, direction: Direction) -> None:
        data = self._post("/move", {"player": player, "direction": direction.value})
        self.current_player = data["current_player"]
        self._winner = data["winner"]

    def is_valid_wall_placement(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> bool:
        result = self._get("/valid-wall", {
            "player": str(player), "orientation": orientation.value,
            "row": str(row), "col": str(col),
        })
        return bool(result["valid"])

    def place_wall(
        self, player: int, orientation: WallOrientation, row: int, col: int
    ) -> None:
        data = self._post("/wall", {
            "player": player, "orientation": orientation.value, "row": row, "col": col,
        })
        self.current_player = data["current_player"]
        self._winner = data["winner"]


if TYPE_CHECKING:
    from quoridor.engine import QuoridorEngine

    def _protocol_conformance_check() -> None:
        _local: EngineLike = QuoridorEngine()
        _remote: EngineLike = RemoteEngine("http://localhost:8765")
