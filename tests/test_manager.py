from __future__ import annotations

import json
import threading
import time
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from vllm_sleeper_proxy.client import HttpResponse
from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager, UnknownModelError, WakeError


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.sleeping = True

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.calls.append((method, url, body))
        if url.endswith("/wake_up") or url.endswith("/wake_up?tags=weights"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/wake_up?tags=kv_cache"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/collective_rpc"):
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/reset_mm_cache"):
            return HttpResponse(200, {}, b"{}")
        if "/sleep?" in url:
            self.sleeping = True
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/is_sleeping"):
            return HttpResponse(200, {}, json.dumps({"is_sleeping": self.sleeping}).encode())
        if url.endswith("/v1/models"):
            return HttpResponse(
                200,
                {},
                b'{"object":"list","data":[{"id":"Qwen/Qwen3-Embedding-8B"}]}',
            )
        if url.endswith("/v1/embeddings"):
            return HttpResponse(200, {}, b'{"object":"list","data":[]}')
        raise AssertionError(f"unexpected request: {method} {url}")


def qwen_model() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-Embedding-8B",
        upstream_model="Qwen/Qwen3-Embedding-8B",
        upstream_base_url="http://vllm:8888/v1",
        control_base_url="http://vllm:8888",
    )


def vision_model() -> ModelConfig:
    return ModelConfig(
        name="chess-vlm-bootstrap",
        upstream_model="Qwen/Qwen3-VL-4B-Instruct",
        upstream_base_url="http://vision:8000/v1",
        control_base_url="http://vision:8000",
    )


class SwitchingHttp(FakeHttp):
    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.calls.append((method, url, body))
        if "/sleep?" in url:
            self.sleeping = True
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/wake_up?tags=weights"):
            self.sleeping = False
            return HttpResponse(200, {}, b"{}")
        if (
            url.endswith("/wake_up?tags=kv_cache")
            or url.endswith("/collective_rpc")
            or url.endswith("/reset_mm_cache")
        ):
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/is_sleeping"):
            return HttpResponse(200, {}, json.dumps({"is_sleeping": self.sleeping}).encode())
        if url.endswith("/v1/models"):
            model = (
                "Qwen/Qwen3-VL-4B-Instruct"
                if url.startswith("http://vision")
                else "Qwen/Qwen3-Embedding-8B"
            )
            return HttpResponse(
                200,
                {},
                json.dumps({"data": [{"id": model}]}).encode(),
            )
        raise AssertionError(f"unexpected request: {method} {url}")


class PerModelSleepHttp(FakeHttp):
    def __init__(self, states: dict[str, bool | None]) -> None:
        super().__init__()
        self.states = states

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.calls.append((method, url, body))
        model = "vision" if url.startswith("http://vision") else "embedding"
        if url.endswith("/is_sleeping"):
            state = self.states[model]
            if state is None:
                return HttpResponse(200, {}, b'{"status":"unknown"}')
            return HttpResponse(200, {}, json.dumps({"is_sleeping": state}).encode())
        if "/sleep?" in url:
            self.states[model] = True
            return HttpResponse(200, {}, b"{}")
        if "wake_up" in url:
            self.states[model] = False
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/collective_rpc") or url.endswith("/reset_mm_cache"):
            return HttpResponse(200, {}, b"{}")
        if url.endswith("/v1/models"):
            upstream = (
                "Qwen/Qwen3-VL-4B-Instruct" if model == "vision"
                else "Qwen3-Embedding-8B"
            )
            return HttpResponse(200, {}, json.dumps({"data": [{"id": upstream}]}).encode())
        raise AssertionError(f"unexpected request: {method} {url}")


class ModelManagerTests(unittest.TestCase):
    def test_openai_model_list_is_logical_and_stable(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        payload = manager.list_openai_models()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "Qwen3-Embedding-8B")

    def test_ensure_awake_wakes_then_checks_readiness(self) -> None:
        http = FakeHttp()
        manager = ModelManager([qwen_model()], http, poll_interval_s=0, wake_timeout_s=1)
        target = manager.ensure_awake("Qwen3-Embedding-8B")
        self.assertEqual(target.upstream_model, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        urls = [url for _, url, _ in http.calls]
        self.assertIn("http://vllm:8888/wake_up?tags=weights", urls)
        self.assertIn("http://vllm:8888/collective_rpc", urls)
        self.assertIn("http://vllm:8888/wake_up?tags=kv_cache", urls)
        self.assertIn("http://vllm:8888/reset_mm_cache", urls)
        self.assertIn("http://vllm:8888/is_sleeping", urls)
        self.assertIn("http://vllm:8888/v1/models", urls)
        self.assertLess(
            urls.index("http://vllm:8888/wake_up?tags=kv_cache"),
            urls.index("http://vllm:8888/reset_mm_cache"),
        )

    def test_unknown_active_state_does_not_wake_when_already_awake(self) -> None:
        http = FakeHttp()
        http.sleeping = False
        manager = ModelManager([qwen_model()], http, poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(manager.ensure_awake("Qwen3-Embedding-8B").name, "Qwen3-Embedding-8B")
        urls = [url for _, url, _ in http.calls]
        self.assertIn("http://vllm:8888/is_sleeping", urls)
        self.assertNotIn("http://vllm:8888/wake_up?tags=weights", urls)
        self.assertNotIn("http://vllm:8888/collective_rpc", urls)

    def test_startup_reconciliation_sleeps_every_awake_model(self) -> None:
        http = PerModelSleepHttp({"embedding": False, "vision": False})
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )

        manager.reconcile_startup_state()

        self.assertEqual(http.states, {"embedding": True, "vision": True})
        self.assertIsNone(manager.active_model_name)
        sleep_urls = [url for method, url, _ in http.calls if method == "POST"]
        self.assertEqual(
            sleep_urls,
            [
                "http://vllm:8888/sleep?level=2",
                "http://vision:8000/sleep?level=2",
            ],
        )

    def test_startup_reconciliation_leaves_sleeping_models_untouched(self) -> None:
        http = PerModelSleepHttp({"embedding": True, "vision": True})
        manager = ModelManager([qwen_model(), vision_model()], http)

        manager.reconcile_startup_state()

        self.assertIsNone(manager.active_model_name)
        self.assertFalse(any(method == "POST" for method, _, _ in http.calls))

    def test_startup_reconciliation_fails_when_sleep_state_is_unknown(self) -> None:
        http = PerModelSleepHttp({"embedding": True, "vision": None})
        manager = ModelManager([qwen_model(), vision_model()], http)

        with self.assertRaisesRegex(
            WakeError,
            "cannot verify startup sleep state for chess-vlm-bootstrap",
        ):
            manager.reconcile_startup_state()

        self.assertIsNone(manager.active_model_name)

    def test_startup_reconciliation_fails_when_engine_stays_awake(self) -> None:
        class RefusesToSleep(PerModelSleepHttp):
            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if "/sleep?" in url:
                    self.calls.append((method, url, body))
                    return HttpResponse(200, {}, b"{}")
                return super().request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )

        http = RefusesToSleep({"embedding": False, "vision": True})
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=0,
        )

        with self.assertRaisesRegex(
            WakeError,
            "timed out waiting for Qwen3-Embedding-8B to enter sleep state",
        ):
            manager.reconcile_startup_state()

        self.assertIsNone(manager.active_model_name)

    def test_startup_reconciliation_fails_on_sleep_state_http_error(self) -> None:
        class SleepStateError(PerModelSleepHttp):
            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if url.endswith("/is_sleeping"):
                    self.calls.append((method, url, body))
                    return HttpResponse(503, {}, b"engine unavailable")
                return super().request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )

        manager = ModelManager(
            [qwen_model()],
            SleepStateError({"embedding": True}),
        )

        with self.assertRaisesRegex(
            WakeError,
            "sleep-state check failed for Qwen3-Embedding-8B: HTTP 503",
        ):
            manager.reconcile_startup_state()

    def test_aliases_resolve_to_model(self) -> None:
        model = ModelConfig(
            name="Qwen3-Embedding-8B",
            aliases=("qwen-emb",),
            upstream_model="Qwen/Qwen3-Embedding-8B",
            upstream_base_url="http://vllm:8888/v1",
            control_base_url="http://vllm:8888",
        )
        manager = ModelManager([model], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(manager.ensure_awake("qwen-emb").name, "Qwen3-Embedding-8B")

    def test_upstream_model_id_resolves_to_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(
            manager.ensure_awake("Qwen/Qwen3-Embedding-8B").name,
            "Qwen3-Embedding-8B",
        )

    def test_litellm_hosted_vllm_model_id_resolves_to_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp(), poll_interval_s=0, wake_timeout_s=1)
        self.assertEqual(
            manager.ensure_awake("hosted_vllm/Qwen/Qwen3-Embedding-8B").name,
            "Qwen3-Embedding-8B",
        )

    def test_unknown_model_is_rejected(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        with self.assertRaises(UnknownModelError):
            manager.ensure_awake("not-a-model")

    def test_admission_denial_happens_before_wake(self) -> None:
        http = FakeHttp()

        def deny(model) -> None:
            raise RuntimeError(f"denied {model.name}")

        manager = ModelManager([qwen_model()], http, admission_check=deny)
        with self.assertRaisesRegex(RuntimeError, "denied Qwen3-Embedding-8B"):
            manager.acquire("Qwen3-Embedding-8B")
        self.assertFalse(
            any("/wake_up" in url for _, url, _ in http.calls)
        )

    def test_resource_backoff_sleeps_active_model(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.ensure_awake("Qwen3-Embedding-8B")

        self.assertEqual("Qwen3-Embedding-8B", manager.sleep_active_model())
        self.assertIsNone(manager.active_model_name)
        self.assertIn(
            "http://vllm:8888/sleep?level=2",
            [url for _, url, _ in http.calls],
        )

    def test_admission_is_rechecked_after_switch_sleep_before_wake(self) -> None:
        http = SwitchingHttp()
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        lease = manager.acquire("Qwen3-Embedding-8B")
        lease.release()
        checks = 0

        def admission(model) -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError("fresh admission denied")

        manager.admission_check = admission
        with self.assertRaisesRegex(RuntimeError, "fresh admission denied"):
            manager.acquire("chess-vlm-bootstrap")
        self.assertFalse(
            any(url.endswith("/wake_up?tags=weights") and url.startswith("http://vision")
                for _, url, _ in http.calls)
        )
        self.assertIsNone(manager.active_model_name)

    def test_request_body_is_rewritten_to_upstream_model(self) -> None:
        manager = ModelManager([qwen_model()], FakeHttp())
        rewritten = manager.rewrite_request_body(
            b'{"model":"Qwen3-Embedding-8B","input":"hello"}', qwen_model()
        )
        self.assertEqual(
            json.loads(rewritten.decode()),
            {"model": "Qwen/Qwen3-Embedding-8B", "input": "hello"},
        )

    def test_model_switch_waits_for_inflight_request(self) -> None:
        http = SwitchingHttp()
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        embedding_lease = manager.acquire("Qwen3-Embedding-8B")
        acquired_vision = threading.Event()

        def switch_model() -> None:
            with manager.acquire("chess-vlm-bootstrap"):
                acquired_vision.set()

        thread = threading.Thread(target=switch_model)
        thread.start()
        time.sleep(0.05)
        self.assertFalse(acquired_vision.is_set())
        self.assertEqual(manager.inflight_requests, 1)
        self.assertFalse(any("/sleep?" in url for _, url, _ in http.calls))

        embedding_lease.release()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired_vision.is_set())
        self.assertEqual(manager.inflight_requests, 0)
        self.assertEqual(manager.active_model_name, "chess-vlm-bootstrap")
        self.assertTrue(any("/sleep?" in url for _, url, _ in http.calls))
        urls = [url for _, url, _ in http.calls]
        sleep_index = next(index for index, url in enumerate(urls)
                           if url.startswith("http://vllm:8888/sleep?"))
        sleep_state_index = next(index for index, url in enumerate(urls[sleep_index + 1:], sleep_index + 1)
                                 if url == "http://vllm:8888/is_sleeping")
        vision_wake_index = urls.index("http://vision:8000/wake_up?tags=weights")
        self.assertLess(sleep_state_index, vision_wake_index)

    def test_request_ownership_releases_on_exception(self) -> None:
        manager = ModelManager(
            [qwen_model()],
            FakeHttp(),
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            with manager.acquire("Qwen3-Embedding-8B"):
                self.assertEqual(manager.inflight_requests, 1)
                raise RuntimeError("upstream failed")
        self.assertEqual(manager.inflight_requests, 0)

    def test_pending_switch_blocks_late_request_for_current_model(self) -> None:
        http = SwitchingHttp()
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        initial = manager.acquire("Qwen3-Embedding-8B")
        vision_acquired = threading.Event()
        release_vision = threading.Event()
        embedding_acquired = threading.Event()

        def switch_to_vision() -> None:
            with manager.acquire("chess-vlm-bootstrap"):
                vision_acquired.set()
                release_vision.wait(timeout=1)

        def late_embedding() -> None:
            with manager.acquire("Qwen3-Embedding-8B"):
                embedding_acquired.set()

        switch_thread = threading.Thread(target=switch_to_vision)
        switch_thread.start()
        deadline = time.monotonic() + 1
        while manager._pending_switch_name != "chess-vlm-bootstrap":  # noqa: SLF001
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        embedding_thread = threading.Thread(target=late_embedding)
        embedding_thread.start()
        initial.release()
        self.assertTrue(vision_acquired.wait(timeout=1))
        self.assertFalse(embedding_acquired.is_set())
        release_vision.set()
        switch_thread.join(timeout=1)
        embedding_thread.join(timeout=1)
        self.assertTrue(embedding_acquired.is_set())

    def test_switch_drain_has_a_deadline(self) -> None:
        manager = ModelManager(
            [qwen_model(), vision_model()],
            SwitchingHttp(),
            poll_interval_s=0,
            wake_timeout_s=1,
            drain_timeout_s=0,
        )
        lease = manager.acquire("Qwen3-Embedding-8B")
        try:
            with self.assertRaisesRegex(WakeError, "timed out draining"):
                manager.acquire("chess-vlm-bootstrap")
        finally:
            lease.release()

    def test_start_reconciles_other_configured_models_even_with_separate_proxy_state(self) -> None:
        http = PerModelSleepHttp({"embedding": True, "vision": False})
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(
                [qwen_model(), vision_model()],
                http,
                startup_lease_path=f"{directory}/startup.lock",
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            manager.acquire("Qwen3-Embedding-8B").release()
        urls = [url for _, url, _ in http.calls]
        self.assertLess(
            urls.index("http://vision:8000/sleep?level=2"),
            urls.index("http://vllm:8888/wake_up?tags=weights"),
        )

    def test_startup_lease_is_created_and_startup_state_clears_after_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_path = f"{directory}/startup.lock"
            manager = ModelManager(
                [qwen_model()],
                FakeHttp(),
                startup_lease_path=lease_path,
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            self.assertIsNone(manager.starting_model_name)
            manager.acquire("Qwen3-Embedding-8B").release()
            self.assertIsNone(manager.starting_model_name)
            self.assertTrue(Path(lease_path).exists())

    def test_startup_reconciliation_holds_and_releases_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_path = f"{directory}/startup.lock"
            manager = ModelManager(
                [qwen_model()],
                FakeHttp(),
                startup_lease_path=lease_path,
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            manager.reconcile_startup_state()
            self.assertIsNone(manager.starting_model_name)
            self.assertIsNone(manager.active_model_name)
            self.assertTrue(Path(lease_path).exists())

    def test_startup_smoke_and_post_start_admission_run_before_lease_release(self) -> None:
        http = FakeHttp()
        model = replace(
            qwen_model(),
            startup_smoke_path="/embeddings",
            startup_smoke_body={"input": "startup"},
        )
        admissions: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(
                [model],
                http,
                startup_lease_path=f"{directory}/startup.lock",
                admission_check=lambda target: admissions.append(target.name),
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            manager.acquire(model.name).release()
        smoke_urls = [url for _, url, _ in http.calls if url.endswith("/v1/embeddings")]
        self.assertEqual(smoke_urls, ["http://vllm:8888/v1/embeddings"])
        self.assertGreaterEqual(len(admissions), 2)
        self.assertTrue(all(item == model.name for item in admissions))

    def test_startup_state_is_held_until_readiness_validation(self) -> None:
        http = FakeHttp()
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(
                [qwen_model()],
                http,
                startup_lease_path=f"{directory}/startup.lock",
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            seen: list[str | None] = []
            original_wait = manager._wait_until_model_listed

            def observe_readiness(model: ModelConfig) -> None:
                seen.append(manager.starting_model_name)
                original_wait(model)

            manager._wait_until_model_listed = observe_readiness  # type: ignore[method-assign]
            manager.acquire("Qwen3-Embedding-8B").release()
            self.assertEqual(seen, ["Qwen3-Embedding-8B"])
            self.assertIsNone(manager.starting_model_name)

    def test_lifecycle_transport_failure_becomes_wake_error(self) -> None:
        class BrokenLifecycle(FakeHttp):
            def request(self, method, url, *, headers=None, body=None, timeout=None):
                raise ConnectionError("control endpoint unavailable")

        manager = ModelManager([qwen_model()], BrokenLifecycle())
        with self.assertRaisesRegex(WakeError, "lifecycle check failed"):
            manager.acquire("Qwen3-Embedding-8B")

    def test_partial_wake_failure_is_slept_before_another_model_wakes(self) -> None:
        class FailVisionReadinessOnce(SwitchingHttp):
            def __init__(self) -> None:
                super().__init__()
                self.fail_vision_readiness = True

            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if (
                    self.fail_vision_readiness
                    and url == "http://vision:8000/v1/models"
                ):
                    self.calls.append((method, url, body))
                    self.fail_vision_readiness = False
                    raise ConnectionError("vision discovery unavailable")
                return super().request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )

        http = FailVisionReadinessOnce()
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        with self.assertRaisesRegex(WakeError, "vision discovery unavailable"):
            manager.acquire("chess-vlm-bootstrap")
        self.assertIsNone(manager.active_model_name)

        all_urls = [url for _, url, _ in http.calls]
        before_retry = len(http.calls)
        manager.acquire("Qwen3-Embedding-8B").release()
        retry_urls = [url for _, url, _ in http.calls[before_retry:]]
        self.assertIn("http://vision:8000/sleep?level=2", all_urls)
        self.assertIn("http://vllm:8888/wake_up?tags=weights", retry_urls)


if __name__ == "__main__":
    unittest.main()
