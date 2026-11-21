"""A scripted, in-process stand-in for LM Studio.

It speaks just enough of LM Studio's APIs for the app:
  * OpenAI-compatible  POST /v1/chat/completions (tool calling), GET /v1/models
  * LM Studio REST     GET  /api/v1/models, POST /api/v1/models/load|unload

Tests queue the model's next answers with `script(...)`, so every run is deterministic
and needs no GPU. Every chat request is recorded for assertions (system prompt, tools, history).
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["gemma-4-26b-a4b-it", "qwen3.8-27b"]


class FakeLMStudio:
    def __init__(self):
        self.queue = []          # pending scripted assistant messages
        self.requests = []       # recorded /v1/chat/completions payloads
        self.loaded = {"gemma-4-26b-a4b-it"}
        self.fail_count = 0      # simulate an LLM outage: answer the next N completions with HTTP 500
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    # --- control API for tests -------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def reset(self):
        with self.lock:
            self.queue.clear()
            self.requests.clear()
            self.fail_count = 0

    def script(self, *tool_calls, reply="Done."):
        """Queue one turn: optional tool calls [(name, args), ...] followed by a final text reply."""
        with self.lock:
            if tool_calls:
                self.queue.append({"tool_calls": [
                    {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for name, args in tool_calls
                ]})
            self.queue.append({"content": reply})

    def script_tool_loop(self, name, args, rounds):
        """Queue `rounds` consecutive tool-call answers (to exercise MAX_TOOL_ROUNDS)."""
        with self.lock:
            for _ in range(rounds):
                self.queue.append({"tool_calls": [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps(args)}}]})

    def last_system_prompt(self):
        return self.requests[-1]["messages"][0]["content"]

    # --- HTTP handling ---------------------------------------------------------------
    def _next_message(self):
        with self.lock:
            return self.queue.pop(0) if self.queue else {"content": "OK."}

    def _handler(fake):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _send(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [
                        {"id": key, "object": "model", "owned_by": "organization_owner"} for key in fake.loaded
                    ]})
                if self.path == "/api/v1/models":
                    return self._send(200, {"models": [
                        {"type": "llm", "key": key, "display_name": key,
                         "loaded_instances": [{"id": key}] if key in fake.loaded else [],
                         "capabilities": {"vision": True, "trained_for_tool_use": True}}
                        for key in MODELS
                    ]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                payload = self._body()
                if self.path == "/v1/chat/completions":
                    fake.requests.append(payload)
                    if fake.fail_count > 0:
                        fake.fail_count -= 1
                        return self._send(500, {"error": "simulated outage"})
                    msg = fake._next_message()
                    message = {"role": "assistant", "content": msg.get("content")}
                    if msg.get("tool_calls"):
                        message["tool_calls"] = msg["tool_calls"]
                    return self._send(200, {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
                        "created": int(time.time()), "model": payload.get("model", ""),
                        "choices": [{"index": 0, "message": message,
                                     "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    })
                if self.path == "/api/v1/models/unload":
                    fake.loaded.discard(payload.get("instance_id"))
                    return self._send(200, {"status": "unloaded"})
                if self.path == "/api/v1/models/load":
                    if payload.get("model") not in MODELS:
                        return self._send(404, {"error": "model not found"})
                    fake.loaded.add(payload["model"])
                    return self._send(200, {"status": "loaded"})
                self._send(404, {"error": "not found"})

        return Handler
