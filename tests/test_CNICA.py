"""Tests for `CNICA` package."""

from __future__ import annotations

import pytest

from CNICA import example_function


def test_version() -> None:
    from CNICA import __version__

    assert __version__ != "999"


@pytest.fixture
def response() -> tuple[int, int]:
    return 1, 2


def test_example_function(response: tuple[int, int]) -> None:
    expected = 3
    assert example_function(*response) == expected
