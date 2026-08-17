from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar

T = TypeVar("T")


class TimeoutExceededError(Exception):
    """A bounded computation exceeded its time budget.

    Best-effort only: Python cannot forcibly kill a running thread, so on
    timeout this stops *waiting* and raises promptly, but the abandoned
    computation keeps running in the background until it finishes on its
    own. In practice the functions this wraps (pathfinding's BFS and
    connected-component search) are mathematically bounded by board size —
    this exists as defensive infrastructure against a future bug changing
    that invariant, not as a way to interrupt a genuinely runaway loop.
    """


def run_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError:
        raise TimeoutExceededError(f"computation exceeded its {seconds}s timeout") from None
    finally:
        executor.shutdown(wait=False)
