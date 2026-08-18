import threading
from collections.abc import Iterator

import pytest

from quoridor.board import QuoridorBoard
from quoridor.engine import QuoridorEngine
from quoridor.server import GameServer, create_server


def _start_server(player_count: int) -> Iterator[tuple[str, QuoridorEngine]]:
    engine = QuoridorEngine(QuoridorBoard(5, player_count))
    server: GameServer = create_server("127.0.0.1", 0, engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        yield f"http://127.0.0.1:{port}", engine
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def live_server() -> Iterator[tuple[str, QuoridorEngine]]:
    yield from _start_server(player_count=2)


@pytest.fixture
def live_server_4p() -> Iterator[tuple[str, QuoridorEngine]]:
    yield from _start_server(player_count=4)
