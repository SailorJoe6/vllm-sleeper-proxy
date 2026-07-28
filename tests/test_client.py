from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest

from vllm_sleeper_proxy.client import UrllibHttpClient


class UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # pragma: no cover - stdlib hook
        return

    def do_POST(self):  # noqa: N802 - stdlib API
        if self.path == "/error":
            body = b'{"error":"bad request"}'
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(b'data: {"delta":"e2"}\n\n')
        self.wfile.flush()
        if self.path == "/delayed":
            time.sleep(0.2)
        self.wfile.write(b"data: [DONE]\n\n")


class StreamingClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_stream_reads_sse_without_buffering_api(self) -> None:
        response = UrllibHttpClient().stream("POST", f"{self.base_url}/stream", timeout=2)
        try:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["content-type"], "text/event-stream")
            self.assertEqual(
                b"".join(response.iter_chunks(8)),
                b'data: {"delta":"e2"}\n\ndata: [DONE]\n\n',
            )
        finally:
            response.close()

    def test_first_sse_event_is_available_before_stream_ends(self) -> None:
        response = UrllibHttpClient().stream("POST", f"{self.base_url}/delayed", timeout=2)
        try:
            chunks = response.iter_chunks()
            started = time.monotonic()
            first = next(chunks)
            elapsed = time.monotonic() - started
            self.assertIn(b'"delta":"e2"', first)
            self.assertLess(elapsed, 0.15)
            self.assertEqual(b"".join(chunks), b"data: [DONE]\n\n")
        finally:
            response.close()

    def test_stream_exposes_upstream_http_error(self) -> None:
        response = UrllibHttpClient().stream("POST", f"{self.base_url}/error", timeout=2)
        try:
            self.assertEqual(response.status, 400)
            self.assertEqual(
                b"".join(response.iter_chunks()),
                b'{"error":"bad request"}',
            )
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
