"""Local wire-level Gradium mock. Run with uv run tests/gradium_mock.py --port 8765."""

import argparse
import asyncio
import base64
import json
import threading

from websockets.asyncio.server import serve


class MockGradium:
    def __init__(self, *, plans=None, drop_first_at=None, error=None, ready=True,
                 finish=True, delay=0, port=0, error_from=0, http_status=None,
                 stall_setup=None, dangling_last=False, stalled_flush=False,
                 dangling_words=(), flush_delay=0, finish_delay=0, stalled_eos=False):
        self.plans = plans or [[("Hello", 0.0, 2.0), ("world.", 2.0, 4.0)]]
        self.drop_first_at, self.error = drop_first_at, error
        self.error_from, self.http_status = error_from, http_status
        self.handshakes = 0
        self.burst_count = 0
        self.stall_setup = stall_setup or {}
        self.dangling_last = dangling_last
        self.stalled_flush = stalled_flush
        self.dangling_words = set(dangling_words)
        self.flush_delay, self.finish_delay = flush_delay, finish_delay
        self.stalled_eos = stalled_eos
        self.ready, self.finish, self.delay, self.port = ready, finish, delay, port
        self.connections = []
        self.errors = []
        self.started = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        assert self.started.wait(3)
        return self

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/api/speech/asr"

    def _run(self):
        import signal

        signal.pthread_sigmask(signal.SIG_BLOCK,
                               {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1})
        asyncio.run(self._serve())

    async def _serve(self):
        self.stop = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        async with serve(self._handle, "127.0.0.1", self.port, process_request=self._request) as server:
            self.port = server.sockets[0].getsockname()[1]
            self.started.set()
            await self.stop.wait()

    def _request(self, connection, request):
        self.handshakes += 1
        if self.http_status:
            return connection.respond(self.http_status, "injected refusal")

    async def _handle(self, ws):
        index = len(self.connections)
        rec = {"messages": [], "samples": 0, "key": ws.request.headers.get("x-api-key"),
               "nonzero_head": None}
        self.connections.append(rec)
        plan = None
        burst_index = None
        sent = 0
        flush_id = 0
        try:
            setup = json.loads(await ws.recv())
            rec["messages"].append(setup)
            assert ws.request.path == "/api/speech/asr"
            assert setup == {"type": "setup", "model_name": "default", "input_format": "pcm_16000",
                             "json_config": {"language": setup["json_config"]["language"]}}
            assert setup["json_config"]["language"] in {"en", "fr", "any"}
            if self.error and index >= self.error_from:
                await ws.send(json.dumps({"type": "error", "message": self.error[0], "code": self.error[1]}))
                return
            if index in self.stall_setup:
                await asyncio.sleep(self.stall_setup[index])
            if not self.ready:
                await self.stop.wait()
                return
            await ws.send(json.dumps({"type": "ready", "sample_rate": 24000, "frame_size": 1920,
                                      "delay_in_frames": 10}))
            async for raw in ws:
                msg = json.loads(raw)
                rec["messages"].append(msg)
                if self.delay:
                    await asyncio.sleep(self.delay)
                if msg["type"] == "audio":
                    if plan is None:
                        burst_index = self.burst_count
                        self.burst_count += 1
                        plan = self.plans[min(burst_index, len(self.plans) - 1)]
                    pcm = base64.b64decode(msg["audio"], validate=True)
                    assert len(pcm) == 2560
                    if rec["nonzero_head"] is None and any(pcm):
                        rec["nonzero_head"] = pcm[:64]
                    rec["samples"] += len(pcm) // 2
                    duration = rec["samples"] / 16000
                    while sent < len(plan) and plan[sent][2] <= duration + 1e-9:
                        text, start, end, *stream = plan[sent]
                        for fragment in (text if isinstance(text, list) else [text]):
                            await ws.send(json.dumps({"type": "text", "text": fragment,
                                                      "start_s": start, "stream_id": stream[0] if stream else 0}))
                        if not ((self.dangling_last and sent == len(plan) - 1)
                                or sent in self.dangling_words):
                            await ws.send(json.dumps({"type": "end_text", "stop_s": end,
                                                      "stream_id": stream[0] if stream else 0}))
                        sent += 1
                    await ws.send(json.dumps({"type": "step", "total_duration_s": duration,
                                              "vad": [{"horizon_s": 2, "inactivity_prob": 0.9}]}))
                    if burst_index == 0 and self.drop_first_at and duration >= self.drop_first_at:
                        await ws.close(code=1011, reason="injected drop")
                        return
                elif msg["type"] == "flush":
                    assert msg["flush_id"] == flush_id + 1
                    flush_id = msg["flush_id"]
                    if self.flush_delay:
                        await asyncio.sleep(self.flush_delay)
                    if self.stalled_flush:
                        while not self.stop.is_set():
                            await ws.send(json.dumps({"type": "step", "total_duration_s": rec["samples"] / 16000,
                                                      "vad": [{"horizon_s": 2, "inactivity_prob": 0.9}]}))
                            await asyncio.sleep(.01)
                    await ws.send(json.dumps({"type": "flushed", "flush_id": flush_id}))
                elif msg["type"] == "end_of_stream":
                    if self.finish_delay:
                        await asyncio.sleep(self.finish_delay)
                    if self.stalled_eos:
                        while not self.stop.is_set():
                            await ws.send(json.dumps({"type": "step", "total_duration_s": rec["samples"] / 16000}))
                            await asyncio.sleep(.01)
                    if self.finish:
                        await ws.send(json.dumps({"type": "end_of_stream"}))
                        return
                else:
                    raise AssertionError(f"unexpected message {msg}")
        except Exception as error:
            from websockets.exceptions import ConnectionClosed
            if not isinstance(error, ConnectionClosed):
                self.errors.append(error)

    def close(self):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        assert not self.errors, self.errors

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    with MockGradium(port=args.port) as mock:
        print(mock.url, flush=True)
        try:
            mock.thread.join()
        except KeyboardInterrupt:
            pass
