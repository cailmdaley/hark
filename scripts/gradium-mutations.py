"""Repeatable isolated mutation checks; real source and hosted Gradium are untouched."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

G = "hark/gradium.py"
C = "hark/cli.py"
V = "hark/voice.py"
T = "tests/test_gradium.py::"
CT = "tests/test_gradium_cli.py::"
VT = "tests/test_cluster.py::"
GT = "tests/test_gradium_grouping.py::"
PT = "tests/test_gradium_source_phrases.py::"
DEFAULT_HARK_HOME = (Path.home() / ".hark").resolve()
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Each fault has a specific observable check, rather than a whole-suite failure.
CASES = [
    ("default-ear", C, 'return "gradium"\n    return "local"', 'return "local"\n    return "local"', CT + "test_default_ear_follows_importability_and_explicit_choice_is_lazy"),
    ("linux-devices", C, 'if not _device_capture_available() and not (args.phone or args.file):', 'if False and not (args.phone or args.file):', CT + "test_linux_device_modes_fail_fast"),
    ("local-capability", C, 'if args.ear == "local" and _default_ear() != "local":', 'if False and _default_ear() != "local":', CT + "test_unavailable_explicit_local_fails_before_lifecycle"),
    ("mlx-dependency-marker", "pyproject.toml", " ; sys_platform == 'darwin'\",\n    \"numpy", '\",\n    "numpy', VT + "test_dependencies_have_portable_platform_markers"),
    ("device-dependency-marker", "pyproject.toml", '"sounddevice>=0.5.1 ; sys_platform == \'darwin\'"', '"sounddevice>=0.5.1"', VT + "test_dependencies_have_portable_platform_markers"),
    ("portable-file-loader", "hark/capture.py", '        audio = load_audio(self.path)', '        from mlx_audio.stt.utils import load_audio\n        audio = load_audio(self.path)', CT + "test_linux_file_cli_and_enrollment_use_portable_audio_without_mlx"),
    ("portable-enrollment", C, '        samples = load_audio(args.file)', '        from mlx_audio.stt.utils import load_audio\n        samples = load_audio(args.file)', CT + "test_linux_file_cli_and_enrollment_use_portable_audio_without_mlx"),
    ("key-precedence", G, 'key = os.environ.get("GRADIUM_API_KEY")', 'key = None', T + "test_key_environment_precedence_permissions_and_lines"),
    ("key-permissions", G, 'stat.S_IMODE(info.st_mode) != 0o600', 'False', T + "test_key_environment_precedence_permissions_and_lines"),
    ("key-one-line", G, 'if not key or any(c.isspace() for c in key):', 'if not key:', T + "test_key_environment_precedence_permissions_and_lines"),
    ("input-format", G, '"input_format": "pcm_16000"', '"input_format": "pcm_24000"', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("language", G, '"json_config": {"language": self.language}', '"json_config": {"language": "en"}', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("api-key-header", G, 'additional_headers={"x-api-key": self.key}', 'additional_headers={"x-wrong-key": self.key}', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("pcm-endianness", G, 'to_pcm16(frame).tobytes()', 'to_pcm16(frame).byteswap().tobytes()', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("frame-size", G, 'FRAME = 1280', 'FRAME = 640', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("ready-model-rate", G, 'or msg["sample_rate"] <= 0', 'or msg["sample_rate"] != SAMPLE_RATE', T + "test_startup_auth_checked_even_all_silent_and_no_audio_sent"),
    ("quiet-speech", G, 'rms=0.001, preroll=0.32', 'rms=0.01, preroll=0.32', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("silence-gate", G, 'voiced = float(np.sqrt(np.mean(samples * samples))) >= self.rms', 'voiced = True', T + "test_startup_auth_checked_even_all_silent_and_no_audio_sent"),
    ("preroll", G, 'for position, frame, count in pre:', 'for position, frame, count in []:', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("hangover", G, 'hangover=0.8, max_duration=10', 'hangover=0.08, max_duration=10', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("source-clock", G, '(run.source + a - run.cloud) / SAMPLE_RATE', 'a / SAMPLE_RATE', T + "test_phone_socket_padding_through_track_and_sink"),
    ("gap-and-reconnect-clock", G, '(run.source + b - run.cloud) / SAMPLE_RATE', 'b / SAMPLE_RATE', T + "test_reconnect_between_bursts_preserves_discarded_gap"),
    ("live-phrases", G, '>= self.phrase_seconds', '>= 9999', T + "test_word_phrases_are_live_and_dangling_last_word_flushes_at_eos"),
    ("word-spacing", G, '(" " if text and word[0] not in ".,;:!?" else "")', '("" if text and word[0] not in ".,;:!?" else "")', T + "test_word_phrases_are_live_and_dangling_last_word_flushes_at_eos"),
    ("text-fragments", G, 'pending[stream] = (text + msg["text"], start)', 'pending[stream] = (text + " " + msg["text"], start)', T + "test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock"),
    ("dangling-eos", G, 'for stream, (text, start) in pending.items():', 'for stream, (text, start) in {}.items():', T + "test_word_phrases_are_live_and_dangling_last_word_flushes_at_eos"),
    ("flush-id", G, 'msg.get("flush_id") == pending_flush:', 'msg.get("flush_id") == pending_flush + 1:', T + "test_final_segment_is_visible_before_burst_end_or_eos"),
    ("retry-duplicates", G, 'if start >= replay_horizons.get(stream, 0) - 1 / SAMPLE_RATE:', 'if True:', T + "test_drop_replays_only_uncommitted_burst_and_counts_retries"),
    ("happy-path-overlap", G, 'replay_horizons.get(stream, 0)', 'burst.horizons.get(stream, 0)', T + "test_overlapping_happy_path_segments_are_not_suppressed"),
    ("retry-boundary-warning", G, 'log(f"gradium: replay skips segment', 'str(f"gradium: replay skips segment', T + "test_replay_boundary_change_is_suppressed_and_logged"),
    ("retry-seconds", G, '        flushed = asyncio.Event()', '        self.sent_seconds = 0.0\n        flushed = asyncio.Event()', T + "test_drop_replays_only_uncommitted_burst_and_counts_retries"),
    ("retry-count", G, 'for attempt in range(self.retries + 1):', 'for attempt in range(1):', T + "test_handshake_auth_is_terminal_but_refusal_retries"),
    ("auth-terminal", G, '{1002, 1008, 401, 403}', '{1002, 401, 403}', T + "test_failures_are_bounded_and_threads_stop"),
    ("backoff", G, 'await asyncio.sleep(self.backoff * 2**attempt)', 'await asyncio.sleep(0)', T + "test_retries_back_off_before_persistent_failure"),
    ("audio-seconds-bound", G, 'if self.backlog_samples + count <= self.backlog_limit:', 'if True:', T + "test_audio_seconds_backlog_bound_is_explicit"),
    ("recoverable-turn-backlog", G, 'backlog_seconds=120', 'backlog_seconds=2', T + "test_many_short_bursts_survive_retryable_setup_delay"),
    ("flush-progress", G, 'await flushed.wait()', 'await asyncio.wait_for(flushed.wait(), self.timeout)', T + "test_progress_keeps_flush_and_final_alive_beyond_idle_timeout"),
    ("final-progress", G, 'settled = max(settled, min(sent_wire, round(duration * SAMPLE_RATE)))\n                        progress()', 'settled = max(settled, min(sent_wire, round(duration * SAMPLE_RATE)))', T + "test_progress_keeps_flush_and_final_alive_beyond_idle_timeout"),
    ("startup-socket-release", G, '        await ws.close()\n        ws = None\n        self.ready.set()', '        self.ready.set()', T + "test_duration_rotation_keeps_long_speech_bounded_and_contiguous"),
    ("duration-rotation", G, 'self.max_frames = max(1, int(max_duration * SAMPLE_RATE / FRAME))', 'self.max_frames = 100000', T + "test_duration_rotation_keeps_long_speech_bounded_and_contiguous"),
    ("character-rotation", G, 'if characters >= 1200:', 'if characters >= 99999:', T + "test_character_rotation_at_completed_segment_boundary"),
    ("cluster-threshold-direction", V, 'scores[index] < self.threshold', 'scores[index] > self.threshold', VT + "test_cluster_returns_to_centroid_and_short_inherits"),
    ("cluster-centroid", V, '(n * self.centroids[index] + vector) / (n + 1)', 'vector', VT + "test_cluster_returns_to_centroid_and_short_inherits"),
    ("short-inheritance", V, 'if len(samples) < self.minimum_duration * 16000:', 'if False:', VT + "test_cluster_returns_to_centroid_and_short_inherits"),
    ("onnx-threads", V, 'min(4, os.cpu_count() or 1)', 'os.cpu_count() or 1', VT + "test_embedder_caps_cpu_threads"),
    ("warm-before-capture", G, '            self.cluster.warm()', '            pass', T + "test_track_warms_default_embedder_before_capture"),
    ("embedding-fallback", G, '                self.cluster_failed = True', '                raise', T + "test_cluster_failure_preserves_all_text_and_logs_once"),
    ("old-raw-audio", V, '        self.file.seek(lo * 2)', '        self.file.seek(0, 2)', VT + "test_archive_keeps_old_audio_across_model_delay"),
    ("voice-naming", C, '            matcher.finished(tracks_by_name[u.track], u)', '            pass', 'tests/test_cli.py::test_voice_matching_failure_logs_once_and_preserves_lines'),
    ("fixed-call-mic", G, 'slot = self.fixed_speaker or self.last_speaker', 'slot = self.last_speaker', T + "test_fixed_call_mic_label_never_clusters"),
    ("missing-key-ended", C, '                        sink.close(f"ended {datetime.now():%H:%M:%S}")', '                        sink.close("unfinished")', CT + "test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors"),
    ("failure-comment", C, '                            sink.comment("gradium " + " ".join(str(capture_error).split()))', '                            pass', CT + "test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors"),
    ("failed-lifecycle", C, 'lifecycle.update("failed", error=', 'lifecycle.update("ended", error=', CT + "test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors"),
    ("mirror-startup-failure", C, 'if args.ear == "gradium" and mirror_target:', 'if False and mirror_target:', CT + "test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors"),
    ("preserve-completed-on-failure", G, '    def flush(self, force=False):\n        self._collect()', '    def flush(self, force=False):\n        self.check()\n        self._collect()', T + "test_committed_segments_survive_later_failure_without_flush_first"),
    ("metering-default", C, 'dest="gradium_metering", action="store_false"', 'dest="gradium_metering", action="store_false", default=False', CT + "test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors"),
    ("metering-schema", G, 'json.load(response)["remaining_credits"]', 'json.load(response)["credits_left"]', CT + "test_metering_uses_exact_documented_endpoint_schema_and_header"),
    ("metering-endpoint", G, 'https://api.gradium.ai/api/usages/credits', 'https://api.gradium.ai/api/credits', CT + "test_metering_uses_exact_documented_endpoint_schema_and_header"),
    ("calibrated-phrase-default", G, 'realtime=True, phrase_seconds=4.0', 'realtime=True, phrase_seconds=2.0', T + "test_calibrated_default_phrase_and_cluster_minimum"),
    ("calibrated-cluster-minimum", G, 'else OnlineCluster(minimum_duration=4)', 'else OnlineCluster()', T + "test_calibrated_default_phrase_and_cluster_minimum"),
    ("phrase-wall-ceiling", G, 'or max(w[2] for w in words) - min(w[1] for w in words) >= 8', 'or False', T + "test_sparse_phrase_has_eight_second_wall_ceiling"),
    ("session-progress-watchdog", G, 'await asyncio.gather(producer, consumer, monitor)\n            return', 'await asyncio.gather(producer, consumer)\n            return', T + "test_unchanged_step_heartbeats_do_not_prevent_failure_or_leave_thread_alive"),
    ("cancel-flush-wait", G, 'if self.cancel.is_set():\n                    raise GradiumError("cancelled")', 'if False:\n                    raise GradiumError("cancelled")', T + "test_cancel_interrupts_a_sender_waiting_for_flush"),
    ("real-tail-clock", G, 'self.request.append(position, samples, valid)', 'self.request.append(position, samples, FRAME)', T + "test_inferred_tail_end_never_exceeds_unpadded_track"),
    ("grouping-default", G, 'idle_seconds=10, source_span=120', 'idle_seconds=.8, source_span=120', GT + "test_default_phone_gaps_share_request_and_preserve_wav_jsonl_and_disjoint_samples"),
    ("ten-second-default", G, 'hangover=0.8, max_duration=10', 'hangover=0.8, max_duration=30', GT + "test_default_ten_submitted_second_rotation_bounds_continuous_replay"),
    ("source-age-default", G, 'idle_seconds=10, source_span=120', 'idle_seconds=10, source_span=240', GT + "test_default_source_age_caps_sparse_grouped_request_at_120_seconds"),
    ("source-map-boundary", G, '(run.source + b - run.cloud) / SAMPLE_RATE', '(run.source + b - run.cloud + FRAME) / SAMPLE_RATE', GT + "test_cloud_join_boundaries_map_end_left_and_start_right"),
    ("source-gap-embedding", G, 'clips = [self.audio.slice(a, b) for a, b in spans]', 'clips = [self.audio.slice(start, end)]', GT + "test_default_phone_gaps_share_request_and_preserve_wav_jsonl_and_disjoint_samples"),
    ("per-flush-publication", G, '                    await boundary()', '                    pass', GT + "test_default_phone_gaps_share_request_and_preserve_wav_jsonl_and_disjoint_samples"),
    ("idle-watchdog", G, 'if waiting() and time.monotonic() - advanced > self.timeout:', 'if time.monotonic() - advanced > self.timeout:', GT + "test_idle_after_speech_flush_is_healthy_and_eos_gets_fresh_progress_budget"),
    ("idle-closing-budget", G, '            progress()\n            sent_eos.set()', '            sent_eos.set()', GT + "test_idle_after_speech_flush_is_healthy_and_eos_gets_fresh_progress_budget"),
    ("unchanged-heartbeats", G, 'if duration > processed:', 'if duration >= processed:', GT + "test_unchanged_flush_heartbeats_do_not_buy_a_progress_budget"),
    ("idle-input-budget", G, 'if not waiting():', 'if False:', GT + "test_new_audio_after_idle_gets_fresh_progress_budget_without_reconnect"),
    ("phrase-overlap-union", G, 'intervals = _union(span for _, _, _, spans in words for span in spans)', 'intervals = list(span for _, _, _, spans in words for span in spans)', GT + "test_overlapping_word_spans_count_unique_source_samples_for_phrase_and_embedding"),
    ("embedding-overlap-union", G, 'spans = _union(span for _, _, _, intervals in words for span in intervals)', 'spans = list(span for _, _, _, intervals in words for span in intervals)', GT + "test_overlapping_word_spans_count_unique_source_samples_for_phrase_and_embedding"),
    ("cross-gap-duration", G, 'sum(b - a for a, b in intervals) >= self.phrase_seconds', 'end - start >= self.phrase_seconds', PT + "test_cross_gap_phrase_counts_only_submitted_intervals_but_caps_source_wall"),
    ("midrequest-replay-clock", G, 'cloud = self.wire_samples', 'cloud = 0', GT + "test_midrequest_drop_replays_mapping_and_deduplicates_cloud_horizon_across_gap"),
    ("linux-signal-threads", C, 'self.portable = sys.platform != "darwin"', 'self.portable = False', CT + "test_real_cli_subprocess_sigterm_after_blas_threads_writes_ended", "linux"),
    ("meeting-ownership", C, ') if _owns_meeting(args, out) else None)', ') if live else None)', 'tests/test_lifecycle_policy.py::test_unowned_live_capture_preserves_existing_record'),
    ("meeting-owner-claim", C, ') if _owns_meeting(args, out) else None)', ') if False else None)', 'tests/test_lifecycle_policy.py::test_owned_live_capture_records_phases'),
    ("file-meeting-ownership", C, 'return not args.file and (args.launch is not None or (', 'return (args.launch is not None or (', 'tests/test_lifecycle_policy.py::test_file_never_owns_record_even_with_launch_and_meetings_output'),
    ("suite-home-isolation", 'tests/conftest.py', 'monkeypatch.setattr(cli, "HOME", home)', 'monkeypatch.setattr(cli, "HOME", Path.home() / ".hark")', 'tests/test_isolation.py::test_suite_home_and_environment_are_isolated'),
    ("cli-home-guard", 'tests/conftest.py', 'if cli.HOME.resolve() == FORBIDDEN_DEFAULT:', 'if False:', 'tests/test_isolation.py::test_cli_guard_refuses_overridden_default_home'),
    ("cli-frozen-home-guard", 'tests/conftest.py', 'if cli.HOME.resolve() == FORBIDDEN_DEFAULT:', 'if cli.HOME.resolve() == (Path.home() / ".hark").resolve():', 'tests/test_isolation.py::test_cli_guard_protects_original_default_after_home_mock'),
    ("cli-environment-guard", 'tests/conftest.py', 'if Path(os.environ["HARK_DIR"]).expanduser().resolve() == FORBIDDEN_DEFAULT:', 'if False:', 'tests/test_isolation.py::test_cli_guard_refuses_default_environment_in_fake_home'),
    ("smoke-home-guard", 'scripts/gradium-smoke.py', 'if home == (Path.home() / ".hark").resolve():', 'if False:', 'tests/test_isolation.py::test_smoke_refuses_default_home_in_sandbox'),
    ("darwin-write-sandbox", 'scripts/gradium-mutations.py', '\n    return [SANDBOX_EXEC, "-p", profile, *command]', '\n    return list(command)', 'tests/test_mutation_sandbox.py::test_physical_sandbox_blocks_only_fake_home_writes', "darwin"),
]


def sandbox_command(command, protected_root):
    """Deny writes beneath an explicit root when the Darwin sandbox is available."""
    if sys.platform != "darwin" or not os.access(SANDBOX_EXEC, os.X_OK):
        return list(command)
    path = json.dumps(str(Path(protected_root).expanduser().resolve()), ensure_ascii=False)
    profile = f'(version 1)(allow default)(deny file-write* (subpath {path}))'
    return [SANDBOX_EXEC, "-p", profile, *command]


def run_tests(root, targets, timeout=40):
    with tempfile.TemporaryDirectory(prefix="hk-check-") as home:
        if Path(home).resolve() == DEFAULT_HARK_HOME:
            raise RuntimeError("mutation checks must not use the default HARK home")
        env = dict(os.environ, HARK_DIR=home, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1",
                   GRADIUM_API_KEY="mutation-local-only")
        start = time.monotonic()
        try:
            command = sandbox_command([sys.executable, "-m", "pytest", "-q", *targets],
                                      DEFAULT_HARK_HOME)
            result = subprocess.run(command, cwd=root, env=env,
                                    capture_output=True, text=True, timeout=timeout)
            return {"returncode": result.returncode, "seconds": round(time.monotonic() - start, 2),
                    "output": (result.stdout + result.stderr)[-5000:]}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "seconds": round(time.monotonic() - start, 2), "output": "test timed out"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=Path("/tmp/hark-gradium-mutations.json"))
    parser.add_argument("--only", help="comma-separated mutation names")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = args.only.split(",") if args.only else None
    report = {"platform": sys.platform, "python": sys.version, "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "baseline": run_tests(root, ["tests"]), "mutations": []}
    if report["baseline"]["returncode"] == 0:
        for case in CASES:
            name, file, old, new, test, *platform = case
            if selected and name not in selected:
                continue
            entry = {"name": name, "file": file, "test": test, "old": old, "new": new}
            if platform and sys.platform != platform[0]:
                entry.update(status="platform-skipped", required_platform=platform[0])
            elif (root / file).read_text().count(old) != 1:
                entry.update(status="invalid-mutation", occurrences=(root / file).read_text().count(old))
            else:
                with tempfile.TemporaryDirectory(prefix="hark-mutation-") as tmp:
                    copy = Path(tmp)
                    for directory in ["hark", "tests", "scripts"]:
                        shutil.copytree(root / directory, copy / directory,
                                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    shutil.copy(root / "pyproject.toml", copy / "pyproject.toml")
                    target = copy / file
                    target.write_text(target.read_text().replace(old, new, 1))
                    entry.update(run_tests(copy, [test]))
                    entry["status"] = "killed" if entry["returncode"] == 1 else "survived" if entry["returncode"] == 0 else "inconclusive"
            report["mutations"].append(entry)
            print(f"{name}: {entry['status']}", flush=True)
            args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(f"receipts: {args.receipt}")
    return 0 if report["baseline"]["returncode"] == 0 and all(
        m["status"] in {"killed", "platform-skipped"} for m in report["mutations"]) else 1


if __name__ == "__main__":
    sys.exit(main())
