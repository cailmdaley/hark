"""Aqua lifecycle detection uses temporary logs and fake CoreAudio processes only."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hark.aqua import AquaDictation, MAX_LEASE_SECONDS, MAX_LOG_BYTES

NOW = datetime(2026, 11, 5, 23, 20, tzinfo=timezone.utc).timestamp()
BUNDLE = "aquavoice.macOSBridge"
ACTIVE = (BUNDLE,)


def processes(pid=10, running=True):
    return [SimpleNamespace(pid=pid, bundle_id=BUNDLE, running_input=running)]


def record(event, lease=3, at=NOW, scope="DictationOrchestrator"):
    stamp = datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return (f"[{stamp}] [info]  ({scope})           dictation.{event} "
            f"provider=online lease_id={lease} result_source=live\n")


def append(path, text):
    with path.open("a") as stream:
        stream.write(text)


@pytest.fixture
def log_path(tmp_path):
    path = tmp_path / "main.log"
    path.touch()
    return path


def test_idle_bridge_with_open_mic_never_pauses(log_path):
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ()
    append(log_path, record("start_rejected_no_owner"))
    assert reader.read(processes(), NOW) == ()


@pytest.mark.parametrize("start", ["start_accepted", "lock_accepted"])
def test_accepted_start_or_standalone_lock_pauses_even_if_coreaudio_is_idle(log_path, start):
    append(log_path, record(start))
    reader = AquaDictation(log_path)
    assert reader.read(processes(running=False), NOW + 1) == ACTIVE
    assert reader.read(processes(running=False), NOW + 20) == ACTIVE


@pytest.mark.parametrize("end", [
    "completed", "completed_canceled", "completed_offline_edit_suppressed",
    "completed_stale_paste_suppressed", "cancel_accepted", "start_failed",
    "owner_lost_during_session", "stop_requested", "stop_ready",
])
def test_precise_matching_terminal_events_resume(log_path, end):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    append(log_path, record(end, at=NOW + 1))
    assert reader.read(processes(), NOW + 2) == ()


def test_stale_lease_or_stale_completion_name_does_not_clear_new_lease(log_path):
    append(log_path, record("start_accepted", 3) + record("start_accepted", 4, NOW + 1))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW + 2) == ACTIVE
    append(log_path, record("completed", 3, NOW + 3)
           + record("completed_canceled_stale", 4, NOW + 3)
           + record("lifecycle_stale", 4, NOW + 3))
    assert reader.read(processes(), NOW + 4) == ACTIVE
    append(log_path, record("completed", 4, NOW + 5))
    assert reader.read(processes(), NOW + 6) == ()


def test_event_requires_exact_scope_parseable_time_and_numeric_lease(log_path):
    append(log_path, record("start_accepted", scope="OtherComponent")
           + record("start_accepted").replace("lease_id=3", "lease_id=invalid")
           + record("start_accepted").replace("2026-11-05T23:20:00.000Z", "invalid"))
    assert AquaDictation(log_path).read(processes(), NOW) == ()


def test_startup_reconstructs_current_and_rotated_logs(log_path):
    old = log_path.with_name("main.old.log")
    old.write_text(record("start_accepted", 2))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW + 1) == ACTIVE
    # Current-file terminal records override starts in the archive on a fresh read.
    append(log_path, record("completed", 2, NOW + 2))
    assert AquaDictation(log_path).read(processes(), NOW + 3) == ()


def test_rotation_consumes_unread_old_tail_and_follows_new_file(log_path):
    append(log_path, record("start_accepted", 2))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    append(log_path, record("completed", 2, NOW + 1))
    log_path.rename(log_path.with_name("main.old.log"))
    log_path.touch()
    assert reader.read(processes(), NOW + 2) == ()
    append(log_path, record("start_accepted", 3, NOW + 3))
    assert reader.read(processes(), NOW + 4) == ACTIVE
    log_path.rename(log_path.with_name("main.old.log"))
    log_path.write_text(record("completed", 3, NOW + 5))
    assert reader.read(processes(), NOW + 6) == ()


def test_rotation_without_retained_old_file_forgets_unknown_active_state(log_path):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    replacement = log_path.with_name("replacement.log")
    replacement.touch()
    replacement.replace(log_path)
    assert reader.read(processes(), NOW + 1) == ()


def test_partial_lines_wait_for_newline_and_are_read_once(log_path):
    start = record("start_accepted")
    log_path.write_text(start[:-1])
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ()
    assert reader.read(processes(), NOW) == ()
    append(log_path, "\n")
    assert reader.read(processes(), NOW + 1) == ACTIVE
    end = record("completed", at=NOW + 2)
    append(log_path, end[:len(end) // 2])
    assert reader.read(processes(), NOW + 3) == ACTIVE
    append(log_path, end[len(end) // 2:])
    assert reader.read(processes(), NOW + 4) == ()
    assert reader.read(processes(), NOW + 5) == ()


@pytest.mark.parametrize("regrow", [False, True])
def test_in_place_truncation_clears_stale_lease_even_if_file_regrows(log_path, regrow):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    log_path.write_text("irrelevant\n" * (40 if regrow else 1))
    assert reader.read(processes(), NOW + 1) == ()
    append(log_path, record("lock_accepted", 4, NOW + 2))
    assert reader.read(processes(), NOW + 3) == ACTIVE


def test_missing_or_unreadable_current_log_fails_open_and_warns_once(log_path, monkeypatch):
    messages = []
    monkeypatch.setattr("hark.aqua.log", messages.append)
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    real_open = Path.open

    def denied(path, *args, **kwargs):
        if path == log_path:
            raise PermissionError("test access denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    assert reader.read(processes(), NOW + 1) == ()
    assert reader.read(processes(), NOW + 2) == ()
    assert len(messages) == 1 and "unavailable" in messages[0] and "unmuted" in messages[0]
    monkeypatch.setattr(Path, "open", real_open)
    log_path.unlink()
    assert reader.read(processes(), NOW + 3) == ()
    assert len(messages) == 1


def test_missing_log_at_startup_is_open_but_follows_a_later_created_file(log_path, monkeypatch):
    messages = []
    monkeypatch.setattr("hark.aqua.log", messages.append)
    log_path.unlink()
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ()
    log_path.write_text(record("start_accepted", at=NOW + 1))
    assert reader.read(processes(), NOW + 2) == ACTIVE
    assert len(messages) == 1


def test_process_disappearance_and_restart_clear_remembered_lease(log_path):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    assert reader.read([], NOW + 1) == ()
    assert reader.read(processes(pid=11), NOW + 2) == ()
    append(log_path, record("start_accepted", 3, NOW + 3))
    assert reader.read(processes(pid=11), NOW + 4) == ACTIVE
    assert reader.read(processes(pid=12), NOW + 5) == ()


def test_helper_churn_and_partial_coreaudio_observations_do_not_end_a_lease(log_path):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    main = SimpleNamespace(pid=9, bundle_id="com.electron.aqua-voice", running_input=False)
    bridge = processes()[0]
    helper = SimpleNamespace(pid=11, bundle_id="com.electron.aqua-voice.helper", running_input=False)
    assert reader.read([main, bridge], NOW)
    assert reader.read([main, bridge, helper], NOW + 1)
    helper.pid = 12
    assert reader.read([main, bridge, helper], NOW + 2)
    assert reader.read([main, helper], NOW + 3)
    assert reader.read([bridge], NOW + 4)
    assert reader.read([main, bridge], NOW + 5)
    bridge.pid = 13
    assert reader.read([main, bridge], NOW + 6) == ()


def test_process_restart_accepts_a_new_start_already_appended_between_polls(log_path):
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    append(log_path, record("start_accepted", 3, NOW + 2))
    assert reader.read(processes(pid=11), NOW + 3) == ACTIVE
    append(log_path, record("completed", 3, NOW + 1))
    assert reader.read(processes(pid=11), NOW + 4) == ACTIVE


def test_15_minute_limit_is_from_start_not_lock_or_poll_and_warns_once(log_path, monkeypatch):
    messages = []
    monkeypatch.setattr("hark.aqua.log", messages.append)
    append(log_path, record("start_accepted"))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    append(log_path, record("lock_accepted", at=NOW + MAX_LEASE_SECONDS - 1))
    assert reader.read(processes(), NOW + MAX_LEASE_SECONDS - 1) == ACTIVE
    assert reader.read(processes(), NOW + MAX_LEASE_SECONDS) == ()
    assert reader.read(processes(), NOW + MAX_LEASE_SECONDS + 1) == ()
    assert len(messages) == 1 and "15-minute" in messages[0]


def test_startup_reused_lease_id_after_crash_uses_new_acceptance(log_path):
    log_path.write_text(record("start_accepted", at=NOW - MAX_LEASE_SECONDS - 100)
                        + record("start_accepted", at=NOW - 1))
    reader = AquaDictation(log_path)
    assert reader.read(processes(), NOW) == ACTIVE
    assert reader.read(processes(), NOW + MAX_LEASE_SECONDS - 1) == ()


def test_old_or_future_unmatched_starts_do_not_mute(log_path, monkeypatch):
    messages = []
    monkeypatch.setattr("hark.aqua.log", messages.append)
    log_path.write_text(record("start_accepted", at=NOW - MAX_LEASE_SECONDS))
    assert AquaDictation(log_path).read(processes(), NOW) == ()
    log_path.write_text(record("start_accepted", at=NOW + 10))
    assert AquaDictation(log_path).read(processes(), NOW) == ()
    assert len(messages) == 2


def test_old_completed_sessions_do_not_warn_on_startup(log_path, monkeypatch):
    messages = []
    monkeypatch.setattr("hark.aqua.log", messages.append)
    log_path.write_text(record("start_accepted", at=NOW - 10000)
                        + record("completed", at=NOW - 9000))
    assert AquaDictation(log_path).read(processes(), NOW) == ()
    assert messages == []


def test_startup_scan_is_bounded_and_skips_its_partial_first_record(log_path):
    old = log_path.with_name("main.old.log")
    old.write_text("x" * (MAX_LOG_BYTES + 100) + "\n" + record("start_accepted"))
    assert AquaDictation(log_path).read(processes(), NOW) == ACTIVE


def test_non_aqua_process_cannot_keep_a_lease_active(log_path):
    append(log_path, record("start_accepted"))
    other = [SimpleNamespace(pid=10, bundle_id="com.apple.CoreSpeech", running_input=True)]
    assert AquaDictation(log_path).read(other, NOW) == ()
