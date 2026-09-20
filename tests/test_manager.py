from __future__ import annotations

import json
import threading
import time
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import vllm_sleeper_proxy.manager as manager_module
from vllm_sleeper_proxy.client import HttpResponse
from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager, UnknownModelError, WakeError, startup_lease
from vllm_sleeper_proxy.thermal_control import ThermalAction


def thermal_action(
    *,
    action_id: str,
    phase: str,
    drain_deadline_epoch: float,
    sleep_deadline_epoch: float,
    overall_deadline_epoch: float,
) -> ThermalAction:
    release = phase in {"release_authorized", "cutoff_recovery_authorized", "releasing"}
    created = drain_deadline_epoch - 5.0
    return ThermalAction(
        record_revision=2 if release else 1,
        incident_id="incident-manager-test",
        action_id=action_id,
        generation=1,
        predecessor_action_id=None,
        transition_kind="graceful_hold",
        phase=phase,
        containment_level="graceful",
        created_at_epoch=created,
        phase_updated_at_epoch=created,
        drain_deadline_epoch=drain_deadline_epoch,
        sleep_deadline_epoch=sleep_deadline_epoch,
        overall_deadline_epoch=overall_deadline_epoch,
        release_authorized_at_epoch=created if release else None,
        repair_deadline_epoch=time.time() + 60 if release else None,
        recovery_authorized=release,
        engine_keys=("Qwen3-Embedding-8B",),
        proof_engine_keys=("Qwen3-Embedding-8B",),
    )


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


def flash_model() -> ModelConfig:
    return ModelConfig(
        name="qwen3.8-flash-next",
        upstream_model="qwen3.8-flash-next",
        upstream_base_url="http://flash:8000/v1",
        control_base_url="http://flash:8000",
    )


class VanishingOwnerHttp(FakeHttp):
    def __init__(self) -> None:
        super().__init__()
        self.owner_unreachable = False
        self.timeouts: list[float | None] = []

    def request(self, method, url, *, headers=None, body=None, timeout=None):
        self.timeouts.append(timeout)
        if self.owner_unreachable and (
            url.endswith("/is_sleeping") or url.endswith("/v1/models")
        ):
            self.calls.append((method, url, body))
            raise ConnectionError("owned engine unavailable")
        return super().request(
            method,
            url,
            headers=headers,
            body=body,
            timeout=timeout,
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
        model = (
            "vision" if url.startswith("http://vision")
            else "flash" if url.startswith("http://flash")
            else "embedding"
        )
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
    def test_owner_validation_timeout_must_be_finite_and_positive(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and greater than zero"):
                    ModelManager(
                        [qwen_model()],
                        FakeHttp(),
                        owner_validation_timeout_s=value,
                    )

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

    def test_fast_admission_denial_happens_before_lifecycle_or_inflight(self) -> None:
        http = FakeHttp()
        calls = []
        def deny(model) -> None:
            calls.append(model.name)
            raise RuntimeError("thermal cooldown")
        manager = ModelManager([qwen_model()], http, pre_admission_check=deny)
        with self.assertRaisesRegex(RuntimeError, "thermal cooldown"):
            manager.acquire("Qwen3-Embedding-8B")
        self.assertEqual(["Qwen3-Embedding-8B"], calls)
        self.assertEqual(0, manager.inflight_requests)
        self.assertEqual([], http.calls)

    def test_fast_admission_rechecks_under_condition_before_wake(self) -> None:
        http = FakeHttp()
        calls = []
        def changes_hot(model) -> None:
            calls.append(model.name)
            if len(calls) == 2:
                raise RuntimeError("thermal cooldown")
        manager = ModelManager([qwen_model()], http, pre_admission_check=changes_hot)
        with self.assertRaisesRegex(RuntimeError, "thermal cooldown"):
            manager.acquire("Qwen3-Embedding-8B")
        self.assertEqual(2, len(calls))
        self.assertEqual([], http.calls)

    def test_fast_admission_denies_promptly_while_quiescing(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()], http, drain_timeout_s=30,
            pre_admission_check=lambda model: (_ for _ in ()).throw(
                RuntimeError("thermal cooldown")
            ),
        )
        with manager._condition:
            manager._quiescing = True
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "thermal cooldown"):
            manager.acquire("Qwen3-Embedding-8B")
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual([], http.calls)

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

    def test_sleep_is_idempotent_after_positive_owned_sleep_evidence(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        http.sleeping = True
        sleep_posts_before = sum("/sleep?" in url for _, url, _ in http.calls)

        self.assertEqual(
            "Qwen3-Embedding-8B",
            manager.sleep_active_model(),
        )
        self.assertEqual(
            sleep_posts_before,
            sum("/sleep?" in url for _, url, _ in http.calls),
        )
        self.assertIsNone(manager.active_model_name)
        self.assertEqual(manager.lifecycle_state, "sleeping")

    def test_sleep_preserves_owned_model_when_sleep_state_is_unknown(self) -> None:
        http = VanishingOwnerHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        http.owner_unreachable = True

        with self.assertRaisesRegex(WakeError, "lifecycle state unavailable"):
            manager.sleep_active_model()

        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        self.assertEqual(manager.lifecycle_state, "unknown")
        self.assertFalse(any("/sleep?" in url for _, url, _ in http.calls))

    def test_thermal_sleep_drains_active_lease_and_denies_late_request(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()], http, poll_interval_s=0,
            wake_timeout_s=1, drain_timeout_s=1,
        )
        lease = manager.acquire("Qwen3-Embedding-8B")
        manager.pre_admission_check = lambda model: (_ for _ in ()).throw(
            RuntimeError("thermal cooldown")
        )
        result = []
        thread = threading.Thread(target=lambda: result.append(manager.sleep_active_model()))
        thread.start()
        deadline = time.monotonic() + 1
        while not manager._quiescing and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(manager._quiescing)
        self.assertTrue(thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "thermal cooldown"):
            manager.acquire("Qwen3-Embedding-8B")
        lease.release()
        thread.join(timeout=2)
        self.assertEqual(["Qwen3-Embedding-8B"], result)
        self.assertIsNone(manager.active_model_name)
        self.assertIn("http://vllm:8888/sleep?level=2", [url for _, url, _ in http.calls])

    def test_idempotent_sleep_does_not_block_next_request(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
            drain_timeout_s=0.05,
        )

        self.assertIsNone(manager.sleep_active_model())
        self.assertIn(
            "http://vllm:8888/is_sleeping",
            [url for _, url, _ in http.calls],
        )
        lease = manager.acquire("Qwen3-Embedding-8B")
        lease.release()

        self.assertEqual("Qwen3-Embedding-8B", manager.active_model_name)
        self.assertTrue(any("/wake_up?tags=weights" in url for _, url, _ in http.calls))

    def test_idempotent_sleep_rejects_unknown_unowned_engine_state(self) -> None:
        class UnknownSleep(FakeHttp):
            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if url.endswith("/is_sleeping"):
                    self.calls.append((method, url, body))
                    raise ConnectionError("engine unavailable")
                return super().request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )

        manager = ModelManager([qwen_model()], UnknownSleep())
        with self.assertRaisesRegex(WakeError, "all-engine sleep state unavailable"):
            manager.sleep_active_model()
        self.assertEqual(manager.lifecycle_state, "unknown")

    def test_lifecycle_status_remains_observable_during_weight_reload(self) -> None:
        class BlockingReloadHttp(FakeHttp):
            def __init__(self) -> None:
                super().__init__()
                self.reload_started = threading.Event()
                self.allow_reload = threading.Event()

            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if url.endswith("/collective_rpc"):
                    self.reload_started.set()
                    self.allow_reload.wait(timeout=2)
                return super().request(method, url, headers=headers, body=body, timeout=timeout)

        http = BlockingReloadHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        startup_done = threading.Event()

        def start() -> None:
            manager.acquire("Qwen3-Embedding-8B").release()
            startup_done.set()

        startup_thread = threading.Thread(target=start)
        startup_thread.start()
        self.assertTrue(http.reload_started.wait(timeout=1))

        observed: list[tuple[bool, str | None, str | None]] = []
        status_done = threading.Event()

        def read_status() -> None:
            observed.append(
                (
                    manager.startup_finalized,
                    manager.active_model_name,
                    manager.starting_model_name,
                )
            )
            status_done.set()

        status_thread = threading.Thread(target=read_status)
        status_thread.start()
        self.assertTrue(status_done.wait(timeout=0.1))
        self.assertEqual(
            [(True, "Qwen3-Embedding-8B", "Qwen3-Embedding-8B")],
            observed,
        )

        http.allow_reload.set()
        startup_thread.join(timeout=2)
        status_thread.join(timeout=2)
        self.assertTrue(startup_done.is_set())

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

    def test_startup_lease_serializes_instances_and_recovers_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_path = Path(directory) / "startup.lock"
            lease_path.write_text("pid=stale\n", encoding="utf-8")
            entered = threading.Event()
            release = threading.Event()

            def holder() -> None:
                with startup_lease(str(lease_path)):
                    entered.set()
                    release.wait(timeout=2)

            thread = threading.Thread(target=holder)
            thread.start()
            self.assertTrue(entered.wait(timeout=1))
            contender_entered = threading.Event()

            def contender() -> None:
                with startup_lease(str(lease_path)):
                    contender_entered.set()

            contender_thread = threading.Thread(target=contender)
            contender_thread.start()
            self.assertFalse(contender_entered.wait(timeout=0.05))
            release.set()
            contender_thread.join(timeout=1)
            thread.join(timeout=1)
            self.assertTrue(contender_entered.is_set())

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

    def test_sleep_waits_until_startup_readiness_finishes(self) -> None:
        class BlockingReadinessHttp(FakeHttp):
            def __init__(self) -> None:
                super().__init__()
                self.readiness_started = threading.Event()
                self.allow_readiness = threading.Event()

            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if url.endswith("/v1/models"):
                    self.readiness_started.set()
                    self.allow_readiness.wait(timeout=2)
                return super().request(method, url, headers=headers, body=body, timeout=timeout)

        http = BlockingReadinessHttp()
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(
                [qwen_model()],
                http,
                startup_lease_path=f"{directory}/startup.lock",
                poll_interval_s=0,
                wake_timeout_s=1,
            )
            startup_done = threading.Event()
            sleeper_done = threading.Event()

            def start() -> None:
                manager.acquire("Qwen3-Embedding-8B").release()
                startup_done.set()

            def sleep() -> None:
                manager.sleep_active_model()
                sleeper_done.set()

            startup_thread = threading.Thread(target=start)
            startup_thread.start()
            self.assertTrue(http.readiness_started.wait(timeout=1))
            sleep_thread = threading.Thread(target=sleep)
            sleep_thread.start()
            self.assertFalse(sleeper_done.wait(timeout=0.05))
            self.assertFalse(any(url.endswith("/sleep?level=2") for _, url, _ in http.calls))
            http.allow_readiness.set()
            startup_thread.join(timeout=2)
            sleep_thread.join(timeout=2)
            self.assertTrue(startup_done.is_set())
            self.assertTrue(sleeper_done.is_set())

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

    def test_cached_ready_owner_is_revalidated_before_same_model_fast_path(self) -> None:
        http = VanishingOwnerHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
            owner_validation_timeout_s=0.1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")

        http.owner_unreachable = True
        with self.assertRaisesRegex(WakeError, "lifecycle state unavailable"):
            manager.acquire("Qwen3-Embedding-8B")

        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        self.assertEqual(manager.lifecycle_state, "unknown")
        self.assertFalse(manager.inference_ready)
        self.assertEqual(manager.inflight_requests, 0)
        self.assertGreater(http.timeouts[-1], 0)
        self.assertLessEqual(http.timeouts[-1], 0.1)

    def test_cached_owner_requires_positive_peer_sleep_evidence(self) -> None:
        for peer_state in (False, None):
            with self.subTest(peer_state=peer_state):
                http = PerModelSleepHttp(
                    {"embedding": False, "vision": peer_state}
                )
                manager = ModelManager(
                    [qwen_model(), vision_model()],
                    http,
                    owner_validation_timeout_s=0.1,
                )
                manager._active_model_name = "Qwen3-Embedding-8B"  # noqa: SLF001
                manager._active_model_ready = True  # noqa: SLF001
                manager._lifecycle_state = "active"  # noqa: SLF001

                with self.assertRaisesRegex(WakeError, "lifecycle state unavailable"):
                    manager.acquire("Qwen3-Embedding-8B")

                mutation_urls = [
                    url
                    for method, url, _ in http.calls
                    if method == "POST" or "/wake_up" in url
                ]
                self.assertEqual(mutation_urls, [])
                self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
                self.assertEqual(manager.lifecycle_state, "unknown")

    def test_unknown_owner_recovers_only_when_every_peer_is_sleeping(self) -> None:
        http = PerModelSleepHttp({"embedding": False, "vision": True})
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            owner_validation_timeout_s=0.1,
        )
        manager._active_model_name = "Qwen3-Embedding-8B"  # noqa: SLF001
        manager._active_model_ready = False  # noqa: SLF001
        manager._lifecycle_state = "unknown"  # noqa: SLF001

        manager.acquire("Qwen3-Embedding-8B").release()

        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        self.assertEqual(manager.lifecycle_state, "active")
        self.assertEqual(
            [],
            [url for method, url, _ in http.calls if method == "POST"],
        )

    def test_cached_ready_owner_found_sleeping_reenters_guarded_wake(self) -> None:
        http = FakeHttp()
        manager = ModelManager(
            [qwen_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        first_wakes = sum("/wake_up" in url for _, url, _ in http.calls)

        http.sleeping = True
        manager.acquire("Qwen3-Embedding-8B").release()

        self.assertGreater(
            sum("/wake_up" in url for _, url, _ in http.calls),
            first_wakes,
        )
        self.assertEqual(manager.lifecycle_state, "active")
        self.assertTrue(manager.inference_ready)

    def test_unknown_owner_blocks_switch_without_waking_second_model(self) -> None:
        class VanishedEmbedding(SwitchingHttp):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_unreachable = False

            def request(self, method, url, *, headers=None, body=None, timeout=None):
                if self.embedding_unreachable and url.startswith("http://vllm"):
                    self.calls.append((method, url, body))
                    raise ConnectionError("owned embedding engine unavailable")
                return super().request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )

        http = VanishedEmbedding()
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            poll_interval_s=0,
            wake_timeout_s=1,
        )
        manager.acquire("Qwen3-Embedding-8B").release()
        http.embedding_unreachable = True
        call_count = len(http.calls)

        with self.assertRaisesRegex(WakeError, "lifecycle state unavailable"):
            manager.acquire("chess-vlm-bootstrap")

        retry_urls = [url for _, url, _ in http.calls[call_count:]]
        self.assertFalse(any(url.startswith("http://vision") and "/wake_up" in url for url in retry_urls))
        self.assertEqual(manager.active_model_name, "Qwen3-Embedding-8B")
        self.assertEqual(manager.lifecycle_state, "unknown")

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


    def test_configured_engine_proof_identities_must_be_unique(self) -> None:
        duplicate_name = replace(vision_model(), name=qwen_model().name)
        duplicate_control = replace(
            vision_model(), control_base_url=qwen_model().control_base_url
        )
        oversized_name = replace(qwen_model(), name="x" * 257)
        for models, message in (
            ([qwen_model(), duplicate_name], "model names must be unique"),
            ([qwen_model(), duplicate_control], "control URLs must be unique"),
            ([oversized_name], "printable and at most 256"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ModelManager(models, PerModelSleepHttp({}))

    def test_thermal_hold_rejects_non_level_two_without_engine_contact(self) -> None:
        http = PerModelSleepHttp({"embedding": False})
        manager = ModelManager([qwen_model()], http, sleep_level=1)
        now = time.time()
        action = thermal_action(
            action_id="thermal-level",
            phase="graceful_hold",
            drain_deadline_epoch=now + 5,
            sleep_deadline_epoch=now + 10,
            overall_deadline_epoch=now + 10,
        )
        with self.assertRaisesRegex(WakeError, "sleep level 2"):
            manager.thermal_hold(action, reauthorize=lambda: None)
        self.assertEqual([], http.calls)

    def test_queued_switch_cannot_wake_after_thermal_hold_proof(self) -> None:
        http = PerModelSleepHttp({"embedding": False, "vision": True})
        fenced = threading.Event()

        def pre_admission(_model: ModelConfig) -> None:
            if fenced.is_set():
                raise WakeError("thermal fence active")

        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            pre_admission_check=pre_admission,
            poll_interval_s=0,
            request_timeout_s=1,
            owner_validation_timeout_s=1,
        )
        manager._active_model_name = qwen_model().name
        manager._active_model_ready = True
        manager._lifecycle_state = "active"
        manager._inflight_requests = 1
        waiter_errors: list[BaseException] = []
        hold_results: list[dict[str, object]] = []

        def wait_for_vision() -> None:
            try:
                manager.acquire(vision_model().name)
            except BaseException as exc:
                waiter_errors.append(exc)

        waiter = threading.Thread(target=wait_for_vision)
        waiter.start()
        deadline = time.monotonic() + 2
        while manager._pending_switch_name != vision_model().name:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)

        fenced.set()
        now = time.time()
        action = thermal_action(
            action_id="thermal-race",
            phase="graceful_hold",
            drain_deadline_epoch=now + 5,
            sleep_deadline_epoch=now + 10,
            overall_deadline_epoch=now + 10,
        )
        holder = threading.Thread(
            target=lambda: hold_results.append(
                manager.thermal_hold(action, reauthorize=lambda: None)
            )
        )
        holder.start()
        deadline = time.monotonic() + 2
        while not manager._quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        manager.release(qwen_model())
        holder.join(timeout=2)
        waiter.join(timeout=2)

        self.assertFalse(holder.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertTrue(hold_results[0]["all_sleeping"])
        self.assertEqual(1, len(waiter_errors))
        self.assertRegex(str(waiter_errors[0]), "thermal fence active")
        self.assertTrue(http.states["embedding"])
        self.assertTrue(http.states["vision"])
        self.assertFalse(any("wake_up" in url for _, url, _ in http.calls))
        self.assertIsNone(manager._pending_switch_name)

    def test_thermal_hold_sleeps_and_positively_proves_every_engine(self) -> None:
        http = PerModelSleepHttp({
            "embedding": False, "vision": True, "flash": False
        })
        manager = ModelManager(
            [qwen_model(), vision_model(), flash_model()],
            http,
            poll_interval_s=0,
            request_timeout_s=1,
            owner_validation_timeout_s=1,
        )
        now = time.time()
        action = thermal_action(
            action_id="thermal-7",
            phase="graceful_hold",
            drain_deadline_epoch=now + 5,
            sleep_deadline_epoch=now + 10,
            overall_deadline_epoch=now + 10,
        )
        proof = manager.thermal_hold(action, reauthorize=lambda: None)
        self.assertTrue(proof["all_sleeping"])
        self.assertEqual(
            {"Qwen3-Embedding-8B", "chess-vlm-bootstrap", "qwen3.8-flash-next"},
            set(proof["engine_proofs"]),
        )
        self.assertTrue(all(item == {
            "state": "sleeping", "proof": "vllm_is_sleeping_true"
        } for item in proof["engine_proofs"].values()))
        self.assertEqual(
            [
                "http://vllm:8888/sleep?level=2",
                "http://flash:8000/sleep?level=2",
            ],
            [url for method, url, _ in http.calls if method == "POST"],
        )

        http.calls.clear()
        replay = manager.thermal_hold(action, reauthorize=lambda: None)
        self.assertTrue(replay["all_sleeping"])
        self.assertFalse(any(method == "POST" for method, _, _ in http.calls))

    def test_generic_sleep_cannot_race_an_action_quiesce(self) -> None:
        http = PerModelSleepHttp({"embedding": True, "vision": True})
        manager = ModelManager([qwen_model(), vision_model()], http)
        manager._quiescing = True
        with self.assertRaisesRegex(WakeError, "already quiescing"):
            manager.sleep_active_model()
        self.assertEqual([], http.calls)

    def test_thermal_hold_expired_deadline_never_contacts_engines(self) -> None:
        http = PerModelSleepHttp({"embedding": False, "vision": True})
        manager = ModelManager([qwen_model(), vision_model()], http)
        now = time.time()
        action = thermal_action(
            action_id="thermal-expired",
            phase="graceful_hold",
            drain_deadline_epoch=now - 3,
            sleep_deadline_epoch=now - 2,
            overall_deadline_epoch=now - 1,
        )
        with self.assertRaisesRegex(WakeError, "overall deadline expired"):
            manager.thermal_hold(action, reauthorize=lambda: None)
        self.assertEqual([], http.calls)

    def test_thermal_hold_rejects_unknown_or_held_awake_state(self) -> None:
        now = time.time()
        for phase, state in (("graceful_hold", None), ("held", False)):
            with self.subTest(phase=phase, state=state):
                http = PerModelSleepHttp({"embedding": True, "vision": state})
                manager = ModelManager(
                    [qwen_model(), vision_model()], http, poll_interval_s=0
                )
                action = thermal_action(
                    action_id="thermal-8",
                    phase=phase,
                    drain_deadline_epoch=now + 5,
                    sleep_deadline_epoch=now + 10,
                    overall_deadline_epoch=now + 10,
                )
                with self.assertRaises(WakeError):
                    manager.thermal_hold(action, reauthorize=lambda: None)
                self.assertFalse(any(method == "POST" for method, _, _ in http.calls))

    def test_thermal_hold_rechecks_authority_before_mutation(self) -> None:
        http = PerModelSleepHttp({"embedding": False, "vision": True})
        manager = ModelManager([qwen_model(), vision_model()], http, poll_interval_s=0)
        now = time.time()
        action = thermal_action(
            action_id="thermal-9",
            phase="graceful_hold",
            drain_deadline_epoch=now + 5,
            sleep_deadline_epoch=now + 10,
            overall_deadline_epoch=now + 10,
        )
        calls = 0

        def changed() -> None:
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise RuntimeError("root phase changed")

        with self.assertRaisesRegex(RuntimeError, "root phase changed"):
            manager.thermal_hold(action, reauthorize=changed)
        self.assertFalse(any(method == "POST" for method, _, _ in http.calls))

    def test_sleeping_subset_proof_contacts_only_exact_retained_peers_and_never_mutates(self) -> None:
        http = PerModelSleepHttp({"embedding": True, "vision": None})
        manager = ModelManager(
            [qwen_model(), vision_model()],
            http,
            owner_validation_timeout_s=1,
        )
        manager._active_model_name = vision_model().name
        manager._active_model_ready = False
        manager._lifecycle_state = "unknown"
        now = time.time()
        action = thermal_action(
            action_id="thermal-subset",
            phase="held",
            drain_deadline_epoch=now - 20,
            sleep_deadline_epoch=now - 10,
            overall_deadline_epoch=now - 5,
        )
        action = replace(
            action,
            engine_keys=(qwen_model().name, vision_model().name),
            proof_engine_keys=(qwen_model().name,),
        )
        proof = manager.thermal_sleeping_subset_proof(
            action, reauthorize=lambda: None
        )
        self.assertTrue(proof["sleeping_subset_verified"])
        self.assertEqual(
            {qwen_model().name}, set(proof["engine_proofs"])
        )
        self.assertTrue(http.calls)
        self.assertTrue(all(
            method == "GET" and url == "http://vllm:8888/is_sleeping"
            for method, url, _ in http.calls
        ))
        self.assertFalse(any(method == "POST" for method, _, _ in http.calls))
        self.assertEqual(vision_model().name, manager.active_model_name)
        self.assertFalse(manager._active_model_ready)
        self.assertEqual("unknown", manager.lifecycle_state)
        self.assertFalse(manager._quiescing)

        awake_http = PerModelSleepHttp({"embedding": False, "vision": None})
        awake_manager = ModelManager(
            [qwen_model(), vision_model()], awake_http,
            owner_validation_timeout_s=1,
        )
        with self.assertRaises(WakeError):
            awake_manager.thermal_sleeping_subset_proof(
                action, reauthorize=lambda: None
            )
        self.assertFalse(any(method == "POST" for method, _, _ in awake_http.calls))
        self.assertFalse(any(url.startswith("http://vision") for _, url, _ in awake_http.calls))
        self.assertFalse(awake_manager._quiescing)

        inflight_http = PerModelSleepHttp({"embedding": True, "vision": None})
        inflight_manager = ModelManager(
            [qwen_model(), vision_model()], inflight_http,
            owner_validation_timeout_s=1,
        )
        inflight_manager._inflight_requests = 1
        with self.assertRaises(WakeError):
            inflight_manager.thermal_sleeping_subset_proof(
                action, reauthorize=lambda: None
            )
        self.assertEqual([], inflight_http.calls)
        self.assertFalse(inflight_manager._quiescing)

    def test_sleeping_subset_proof_budget_bounds_each_shared_lock_stage(self) -> None:
        now = time.time()
        action = thermal_action(
            action_id="thermal-subset-locks",
            phase="held",
            drain_deadline_epoch=now - 20,
            sleep_deadline_epoch=now - 10,
            overall_deadline_epoch=now - 5,
        )
        with tempfile.TemporaryDirectory() as directory:
            for stage in ("action", "condition", "startup"):
                with self.subTest(stage=stage):
                    http = PerModelSleepHttp({"embedding": True})
                    manager = ModelManager(
                        [qwen_model()],
                        http,
                        startup_lease_path=str(Path(directory) / "startup.lock"),
                    )
                    release = threading.Event()
                    holder = None
                    outer_lease = None
                    if stage == "action":
                        manager._thermal_action_lock.acquire()
                    elif stage == "condition":
                        entered = threading.Event()

                        def hold_condition() -> None:
                            with manager._condition:
                                entered.set()
                                release.wait(timeout=1)

                        holder = threading.Thread(target=hold_condition)
                        holder.start()
                        self.assertTrue(entered.wait(timeout=1))
                    else:
                        outer_lease = startup_lease(manager.startup_lease_path)
                        outer_lease.__enter__()
                    original = manager_module.MAX_SLEEPING_SUBSET_PROOF_SECONDS
                    manager_module.MAX_SLEEPING_SUBSET_PROOF_SECONDS = 0.05
                    started = time.monotonic()
                    try:
                        with self.assertRaises(WakeError):
                            manager.thermal_sleeping_subset_proof(
                                action, reauthorize=lambda: None
                            )
                    finally:
                        manager_module.MAX_SLEEPING_SUBSET_PROOF_SECONDS = original
                        if stage == "action":
                            manager._thermal_action_lock.release()
                        if outer_lease is not None:
                            outer_lease.__exit__(None, None, None)
                        release.set()
                        if holder is not None:
                            holder.join(timeout=1)
                    self.assertLess(time.monotonic() - started, 0.5)
                    self.assertEqual([], http.calls)
                    self.assertFalse(manager._quiescing)
                    self.assertTrue(manager._thermal_action_lock.acquire(timeout=0.1))
                    manager._thermal_action_lock.release()

    def test_sleeping_subset_proof_discards_partial_result_on_authority_change(self) -> None:
        http = PerModelSleepHttp({
            "embedding": True, "vision": True, "flash": True
        })
        manager = ModelManager(
            [qwen_model(), vision_model(), flash_model()],
            http,
            owner_validation_timeout_s=1,
        )
        now = time.time()
        action = thermal_action(
            action_id="thermal-subset-change",
            phase="held",
            drain_deadline_epoch=now - 20,
            sleep_deadline_epoch=now - 10,
            overall_deadline_epoch=now - 5,
        )
        action = replace(
            action,
            engine_keys=tuple(sorted((
                qwen_model().name, vision_model().name, flash_model().name
            ))),
            proof_engine_keys=tuple(sorted((
                qwen_model().name, flash_model().name
            ))),
        )
        calls = 0

        def changed() -> None:
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise RuntimeError("root action changed")

        with self.assertRaisesRegex(RuntimeError, "root action changed"):
            manager.thermal_sleeping_subset_proof(
                action, reauthorize=changed
            )
        self.assertFalse(any(method == "POST" for method, _, _ in http.calls))
        self.assertFalse(manager._quiescing)

    def test_thermal_release_is_positive_proof_only_and_never_mutates_engines(self) -> None:
        now = time.time()
        action = thermal_action(
            action_id="thermal-10",
            phase="release_authorized",
            drain_deadline_epoch=now - 20,
            sleep_deadline_epoch=now - 10,
            overall_deadline_epoch=now - 5,
        )
        sleeping_http = PerModelSleepHttp({"embedding": True, "vision": True})
        manager = ModelManager([qwen_model(), vision_model()], sleeping_http)
        proof = manager.thermal_release_ready(action, reauthorize=lambda: None)
        self.assertTrue(proof["release_ready"])
        self.assertFalse(any(method == "POST" for method, _, _ in sleeping_http.calls))

        awake_http = PerModelSleepHttp({"embedding": True, "vision": False})
        manager = ModelManager([qwen_model(), vision_model()], awake_http)
        with self.assertRaises(WakeError):
            manager.thermal_release_ready(action, reauthorize=lambda: None)
        self.assertFalse(any(method == "POST" for method, _, _ in awake_http.calls))


if __name__ == "__main__":
    unittest.main()
