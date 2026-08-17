import time

import pytest

from quoridor.timeouts import TimeoutExceededError, run_with_timeout


def test_returns_value_when_completed_in_time():
    assert run_with_timeout(lambda: 42, seconds=1.0) == 42


def test_raises_when_computation_exceeds_budget():
    def slow() -> int:
        time.sleep(0.2)
        return 1

    with pytest.raises(TimeoutExceededError):
        run_with_timeout(slow, seconds=0.01)


def test_propagates_the_wrapped_function_s_own_exception():
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_with_timeout(boom, seconds=1.0)
