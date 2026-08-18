import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from quoridor.board import QuoridorBoard
from quoridor.engine import Direction, InvalidMoveError, QuoridorEngine, WallOrientation


def _state_payload(engine: QuoridorEngine) -> dict[str, Any]:
    return {
        "board": engine.get_state(),
        "current_player": engine.current_player,
        "winner": engine.winner(),
    }


class GameServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler],
        engine: QuoridorEngine,
    ):
        super().__init__(address, handler_cls)
        self.engine = engine
        self.lock = threading.Lock()
        # Which player numbers already have a connection claiming them —
        # guarded by the same lock as engine state. Seats are never
        # released once claimed (no reconnect story yet); a stuck seat
        # just means restarting the server, matching this server's existing
        # one-game-per-process lifetime.
        self.claimed_players: set[int] = set()


class GameRequestHandler(BaseHTTPRequestHandler):
    @property
    def _server(self) -> GameServer:
        return cast(GameServer, self.server)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return dict(json.loads(raw))

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        engine = self._server.engine

        with self._server.lock:
            try:
                if parsed.path == "/health":
                    self._send_json(200, {"status": "ok"})
                elif parsed.path == "/state":
                    self._send_json(200, _state_payload(engine))
                elif parsed.path == "/valid-move":
                    player = int(query["player"])
                    direction = Direction(query["direction"])
                    self._send_json(200, {"valid": engine.is_valid_move(player, direction)})
                elif parsed.path == "/valid-wall":
                    player = int(query["player"])
                    orientation = WallOrientation(query["orientation"])
                    row = int(query["row"])
                    col = int(query["col"])
                    valid = engine.is_valid_wall_placement(player, orientation, row, col)
                    self._send_json(200, {"valid": valid})
                else:
                    self._send_json(404, {"error": "not found"})
            except (KeyError, ValueError) as e:
                self._send_json(400, {"error": str(e)})

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        engine = self._server.engine

        with self._server.lock:
            try:
                body = self._read_json_body()
                if parsed.path == "/claim":
                    player_count = engine.board.player_count
                    requested = body.get("player")
                    if requested is not None:
                        player = int(requested)
                        if not (1 <= player <= player_count):
                            raise ValueError(f"invalid player: {player}")
                        if player in self._server.claimed_players:
                            self._send_json(
                                409, {"error": f"player {player} is already connected"}
                            )
                            return
                    else:
                        free = [
                            p for p in range(1, player_count + 1)
                            if p not in self._server.claimed_players
                        ]
                        if not free:
                            self._send_json(409, {"error": "game is full"})
                            return
                        player = free[0]
                    self._server.claimed_players.add(player)
                    self._send_json(200, {"player": player})
                elif parsed.path == "/move":
                    player = int(body["player"])
                    direction = Direction(body["direction"])
                    try:
                        engine.move(player, direction)
                    except InvalidMoveError as e:
                        self._send_json(409, {"error": str(e)})
                        return
                    self._send_json(200, _state_payload(engine))
                elif parsed.path == "/wall":
                    player = int(body["player"])
                    orientation = WallOrientation(body["orientation"])
                    row = int(body["row"])
                    col = int(body["col"])
                    try:
                        engine.place_wall(player, orientation, row, col)
                    except InvalidMoveError as e:
                        self._send_json(409, {"error": str(e)})
                        return
                    self._send_json(200, _state_payload(engine))
                else:
                    self._send_json(404, {"error": "not found"})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": str(e)})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def create_server(host: str, port: int, engine: QuoridorEngine) -> GameServer:
    return GameServer((host, port), GameRequestHandler, engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Quoridor game server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--size", type=int, default=9, choices=[5, 7, 9])
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    args = parser.parse_args()

    engine = QuoridorEngine(QuoridorBoard(args.size, args.players))
    server = create_server(args.host, args.port, engine)
    print(
        f"Quoridor server listening on {args.host}:{args.port} "
        f"(board size {args.size}, {args.players} players)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
