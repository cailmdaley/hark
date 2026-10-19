"""Residual echo matching never removes different simultaneous near-end speech."""
from datetime import datetime, timedelta

import pytest

from hark.dedupe import EchoDeduper, echo_match
from hark.transcript import Utterance


def utterance(track, text, start=0, duration=3):
    return Utterance(track, "S1", 0, duration, datetime(2026, 10, 1) + timedelta(seconds=start), text)


@pytest.mark.parametrize("mic,system,match", [
    ("The purple telescope measures galaxies!", "the PURPLE telescope measures galaxies", True),
    ("We need to discuss the covariance matrix today", "We need to discuss a covariance matrix today", True),
    ("I disagree about the covariance", "We need to discuss the covariance matrix today", False),
    ("Yes", "Yes", False),
    ("Please write down forty two but I think it is wrong", "Please write down forty two", False),
])
def test_fuzzy_text(mic, system, match):
    assert echo_match(utterance("mic", mic), utterance("system", system)) == match


def test_identical_nonoverlapping_text_survives():
    assert not echo_match(utterance("mic", "The exact same sentence", start=3),
                          utterance("system", "The exact same sentence"))


@pytest.mark.parametrize("order", ["mic-first", "system-first", "batch"])
def test_both_completion_orders_and_drop_logging(order):
    messages = []
    deduper = EchoDeduper(report=messages.append)
    mic, system = (utterance(track, "The exact same sentence") for track in ("mic", "system"))
    now = mic.wall.timestamp() + 5
    if order == "batch":
        result = deduper.feed([mic, system], now)
    else:
        first, second = (mic, system) if order == "mic-first" else (system, mic)
        result = deduper.feed([first], now) + deduper.feed([second], now + 1)
    result += deduper.feed([], now + 4, final=True)
    assert result == [system]
    assert len(messages) == 1 and "dropped mic echo" in messages[0]


def test_added_near_end_words_and_final_pending_are_preserved():
    deduper = EchoDeduper()
    mic = utterance("mic", "I am speaking at the same time")
    system = utterance("system", "This is different remote speech")
    now = mic.wall.timestamp() + 5
    assert deduper.feed([mic, system], now) == [system]
    assert deduper.feed([], now + 2) == []
    assert deduper.feed([], now + 3) == [mic]
    assert deduper.feed([mic], now + 4, final=True) == [mic]


def test_recent_cache_is_bounded():
    deduper = EchoDeduper()
    system = utterance("system", "This is remote speech")
    now = system.wall.timestamp()
    deduper.feed([system], now)
    deduper.feed([], now + 70)
    assert not deduper.system
