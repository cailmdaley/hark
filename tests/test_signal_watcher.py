import signal
import threading
from types import SimpleNamespace

from hark import cli


def test_portable_idle_close_wakes_and_joins(monkeypatch):
    monkeypatch.setattr(cli.sys, 'platform', 'linux')
    stop = threading.Event()
    watcher = cli.SignalWatcher(None, stop)
    watcher.close()
    assert not watcher.thread.is_alive()
    assert not stop.is_set()


def test_portable_signal_updates_lifecycle_outside_handler_and_second_interrupt_quits(monkeypatch):
    monkeypatch.setattr(cli.sys, 'platform', 'linux')
    updates = []
    exits = []
    stop = threading.Event()
    watcher = cli.SignalWatcher(SimpleNamespace(stopping=lambda: updates.append(threading.get_ident())), stop)
    monkeypatch.setattr(cli.os, '_exit', exits.append)
    try:
        watcher._receive(signal.SIGINT, None)
        assert stop.is_set()
        assert watcher.signal_written.wait(1)
        assert updates == [watcher.thread.ident]
        watcher._receive(signal.SIGINT, None)
        assert exits == [128 + signal.SIGINT]
    finally:
        watcher.close()
    assert not watcher.thread.is_alive()
