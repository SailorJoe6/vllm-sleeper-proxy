from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

from vllm_sleeper_proxy.config import ModelConfig
from vllm_sleeper_proxy.manager import ModelManager, WakeError
from vllm_sleeper_proxy.server import build_server
from vllm_sleeper_proxy.thermal_control import (
    FileThermalActionAuthority,
    ThermalActionControlError,
)


class NoEngineHttp:
    def __init__(self) -> None:
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("projection control must not contact an engine")


def model() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-Embedding-8B",
        upstream_model="Qwen/Qwen3-Embedding-8B",
        upstream_base_url="http://vllm:8888/v1",
        control_base_url="http://vllm:8888",
    )


def inner(now: float, *, active: bool = True) -> dict[str, object]:
    if not active:
        return {
            "schema_version": 2,
            "generated_at_epoch": now - 1,
            "active": False,
            "state": "none",
            "applied": False,
            "action_id": None,
            "reason_codes": [],
            "requirement": "REQ-MODEL-AVAIL-001",
        }
    return {
        "schema_version": 2,
        "record_revision": 1,
        "incident_id": "incident-1",
        "action_id": "thermal-1",
        "generation": 1,
        "predecessor_action_id": None,
        "transition_kind": "graceful_hold",
        "created_at_epoch": now - 2,
        "phase_updated_at_epoch": now - 2,
        "generated_at_epoch": now - 1,
        "active": active,
        "state": "sleep",
        "phase": "graceful_hold",
        "containment_level": "graceful",
        "applied": False,
        "result": "pending",
        "recovery_authorized": False,
        "reason_codes": ["acpi_temperature_requires_sleep"],
        "drain_deadline_epoch": now + 30,
        "sleep_deadline_epoch": now + 60,
        "overall_deadline_epoch": now + 90,
        "release_authorized_at_epoch": None,
        "repair_deadline_epoch": None,
        "repair_target_engine_key": None,
        "repair_target_status": None,
        "repair_target_deadline_epoch": None,
        "engine_keys": ["Qwen3-Embedding-8B"],
        "sleeping_peer_engine_keys": [],
        "authorized_operations": ["hold"] if active else [],
        "requirement": "REQ-MODEL-AVAIL-001",
    }


def digest(value: dict[str, object]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def semantic_id(value: dict[str, object]) -> str:
    semantic = dict(value)
    semantic.pop("publication_id", None)
    semantic.pop("predecessor_publication_id", None)
    semantic.pop("successor_publication_id", None)
    return digest(semantic)


def envelope(
    now: float,
    state: str,
    *,
    inactive: bool = False,
) -> dict[str, object]:
    projection = inner(now, active=not inactive)
    common = {
        "projection_protocol_version": 1,
        "transition": "bootstrap",
        "source_store_revision": 1,
        "source_record_revision": 1,
        "target_store_revision": 1,
        "target_record_revision": 1,
        "source_authority_sha256": None,
        "target_authority_sha256": digest(projection),
        "prepared_at_epoch": now - 1,
        "deadline_epoch": now + 10,
        "requirement": "REQ-MODEL-AVAIL-001",
    }
    final = {
        **common,
        "publication_state": "inactive" if inactive else "active",
        "publication_sequence": 2,
        "predecessor_publication_sequence": 1,
        "publication_id": "0" * 64,
        "predecessor_publication_id": "0" * 64,
        "successor_publication_id": None,
        "authority": projection,
    }
    final_id = semantic_id(final)
    quiescing = {
        **common,
        "publication_state": "quiescing",
        "publication_sequence": 1,
        "predecessor_publication_sequence": None,
        "publication_id": "0" * 64,
        "predecessor_publication_id": None,
        "successor_publication_id": final_id,
        "authority": None,
    }
    quiescing_id = semantic_id(quiescing)
    final["publication_id"] = final_id
    final["predecessor_publication_id"] = quiescing_id
    quiescing["publication_id"] = quiescing_id
    return quiescing if state == "quiescing" else final


def replacement_pair(
    previous: dict[str, object], previous_final: dict[str, object], now: float
) -> tuple[dict[str, object], dict[str, object]]:
    refreshed_authority = dict(previous_final["authority"])
    refreshed_authority["generated_at_epoch"] = now
    refreshed_digest = digest(refreshed_authority)
    final = {
        **previous,
        "publication_state": "active",
        "publication_sequence": int(previous["publication_sequence"]) + 2,
        "predecessor_publication_sequence": int(previous["publication_sequence"]) + 1,
        "publication_id": "0" * 64,
        "predecessor_publication_id": "0" * 64,
        "successor_publication_id": None,
        "target_authority_sha256": refreshed_digest,
        "prepared_at_epoch": now,
        "deadline_epoch": now + 5,
        "authority": refreshed_authority,
    }
    final_id = semantic_id(final)
    replacement = {
        **previous,
        "publication_sequence": int(previous["publication_sequence"]) + 1,
        "predecessor_publication_sequence": previous["publication_sequence"],
        "predecessor_publication_id": previous["publication_id"],
        "publication_id": "0" * 64,
        "successor_publication_id": final_id,
        "target_authority_sha256": refreshed_digest,
        "prepared_at_epoch": now,
        "deadline_epoch": now + 5,
    }
    replacement["publication_id"] = semantic_id(replacement)
    final["publication_id"] = final_id
    final["predecessor_publication_id"] = replacement["publication_id"]
    return replacement, final


def control_request(value: dict[str, object], operation: str) -> dict[str, object]:
    return {"operation": operation, **{key: item for key, item in value.items() if key != "authority"}}


def action_request(
    value: dict[str, object], *,
    proxy_instance_id: str = "0" * 64,
    activation_token: str = "0" * 64,
) -> dict[str, object]:
    projection = value["authority"]
    assert isinstance(projection, dict)
    keys = {
        "record_revision", "incident_id", "action_id", "generation",
        "predecessor_action_id", "transition_kind", "phase",
        "containment_level", "created_at_epoch", "phase_updated_at_epoch",
        "drain_deadline_epoch", "sleep_deadline_epoch", "overall_deadline_epoch",
        "release_authorized_at_epoch", "repair_deadline_epoch",
        "repair_target_engine_key", "repair_target_status",
        "repair_target_deadline_epoch", "recovery_authorized", "engine_keys",
    }
    return {
        "schema_version": 2,
        "operation": "hold",
        **{key: projection[key] for key in keys},
        "proof_engine_keys": list(projection["engine_keys"]),
        "publication_sequence": value["publication_sequence"],
        "publication_id": value["publication_id"],
        "target_store_revision": value["target_store_revision"],
        "target_authority_sha256": value["target_authority_sha256"],
        "proxy_instance_id": proxy_instance_id,
        "activation_token": activation_token,
    }


class ProjectionProtocolTests(unittest.TestCase):
    def write(self, path: Path, value: object) -> None:
        path.write_bytes(json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode())
        path.chmod(0o600)

    def test_strict_envelope_control_action_and_restart_adoption(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing")
            a = envelope(now, "active")
            self.write(path, q)
            authority = FileThermalActionAuthority(
                path, expected_uid=os.geteuid(), expected_mode=0o600,
                allow_legacy=False,
            )
            q_value = authority.authorize_control("quiesce", control_request(q, "quiesce"))
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            self.assertEqual("unknown", manager.projection_quiesce(q_value))
            self.write(path, a)
            a_value = authority.authorize_control("activate", control_request(a, "activate"))
            self.assertEqual("unknown", manager.projection_activate(a_value))
            action = authority.authorize(
                "hold", action_request(
                    a,
                    proxy_instance_id=manager.proxy_instance_id,
                    activation_token=manager.projection_activation_token or "",
                )
            )
            manager.require_projection_action(action)
            restarted = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            restarted.configure_projection_control()
            with self.assertRaises(WakeError):
                restarted.require_projection_action(action)

    def test_inactive_adoption_clears_gate_but_authorizes_no_action(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing", inactive=True)
            inactive = envelope(now, "inactive", inactive=True)
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            self.write(path, q)
            manager.projection_quiesce(
                authority.authorize_control("quiesce", control_request(q, "quiesce"))
            )
            self.write(path, inactive)
            phase = manager.projection_activate(
                authority.authorize_control("activate", control_request(inactive, "activate"))
            )
            self.assertEqual("unknown", phase)
            self.assertTrue(manager.projection_control_ready)
            self.assertEqual("inactive", manager.projection_state)
            manager._require_projection_active_locked()  # gate is intentionally clear
            with self.assertRaises(ThermalActionControlError) as caught:
                authority.authorize("hold", action_request(envelope(now, "active")))
            self.assertEqual("inactive", caught.exception.code)

    def test_transitional_sleeping_waits_but_stable_lifecycle_sleeping_maps_idle(self) -> None:
        now = time.time()
        authority_value = envelope(now, "quiescing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            self.write(path, authority_value)
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            parsed = authority.authorize_control(
                "quiesce", control_request(authority_value, "quiesce")
            )
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            with manager._condition:
                manager._lifecycle_state = "sleeping"
                manager._transition_phase = "sleeping"
            completed = threading.Event()
            result: list[str] = []

            def run() -> None:
                result.append(manager.projection_quiesce(parsed))
                completed.set()

            thread = threading.Thread(target=run)
            thread.start()
            with manager._condition:
                self.assertTrue(manager._condition.wait_for(
                    lambda: manager.projection_state == "quiescing", timeout=1
                ))
                self.assertFalse(completed.is_set())
                manager._transition_phase = "idle"
                manager._condition.notify_all()
            self.assertTrue(completed.wait(2))
            thread.join(timeout=1)
            self.assertEqual(["idle"], result)

    def test_strict_file_identity_type_mode_and_link_count(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "authority.json"
            self.write(path, envelope(now, "active"))
            request = action_request(envelope(now, "active"))
            for authority in (
                FileThermalActionAuthority(path, expected_uid=os.geteuid() + 1, expected_mode=0o600, allow_legacy=False),
                FileThermalActionAuthority(path, expected_uid=os.geteuid(), expected_mode=0o640, allow_legacy=False),
            ):
                with self.assertRaises(ThermalActionControlError):
                    authority.authorize("hold", request)
            link = root / "linked.json"
            os.link(path, link)
            with self.assertRaises(ThermalActionControlError):
                FileThermalActionAuthority(path, allow_legacy=False).authorize("hold", request)
            link.unlink()
            alias = root / "alias.json"
            alias.symlink_to(path)
            with self.assertRaises(ThermalActionControlError):
                FileThermalActionAuthority(alias, allow_legacy=False).authorize("hold", request)

    def test_control_endpoints_return_operation_specific_canonical_ack(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing")
            a = envelope(now, "active")
            self.write(path, q)
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            server = build_server(
                "127.0.0.1", 0, manager,
                thermal_action_control_enabled=True,
                thermal_action_authority=FileThermalActionAuthority(
                    path, allow_legacy=False
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def post(route: str, value: dict[str, object]) -> tuple[bytes, dict[str, object]]:
                encoded = json.dumps(value, separators=(",", ":")).encode()
                request = Request(
                    base + route,
                    data=encoded,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    raw = response.read()
                return raw, json.loads(raw)

            try:
                q_raw, q_ack = post(
                    "/thermal/projection/quiesce", control_request(q, "quiesce")
                )
                self.assertTrue(q_ack.pop("quiesced"))
                self.assertNotIn("activated", q_ack)
                self.assertNotIn("activation_token", q_ack)
                instance = q_ack["proxy_instance_id"]
                self.assertRegex(instance, r"^[0-9a-f]{64}$")
                self.assertEqual(
                    q_raw,
                    json.dumps(
                        {**q_ack, "quiesced": True},
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode(),
                )
                self.assertEqual([], manager.http.calls)
                self.write(path, a)
                a_raw, a_ack = post(
                    "/thermal/projection/activate", control_request(a, "activate")
                )
                self.assertTrue(a_ack.pop("activated"))
                self.assertNotIn("quiesced", a_ack)
                self.assertEqual(instance, a_ack["proxy_instance_id"])
                activation_token = a_ack["activation_token"]
                self.assertRegex(activation_token, r"^[0-9a-f]{64}$")
                self.assertEqual(
                    a_raw,
                    json.dumps(
                        {**a_ack, "activated": True},
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode(),
                )
                self.assertEqual([], manager.http.calls)
                # Exact retries are idempotent and retain the process instance.
                _, retry = post(
                    "/thermal/projection/activate", control_request(a, "activate")
                )
                self.assertEqual(instance, retry["proxy_instance_id"])
                self.assertEqual(activation_token, retry["activation_token"])
                self.assertTrue(retry["activated"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_phase_matrix_and_unbound_normal_lifecycle_are_fail_closed(self) -> None:
        manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
        manager.configure_projection_control()
        with manager._condition:
            for phase in (
                "starting", "sleeping", "finalizing", "adopting",
                "bootstrap_sleep",
            ):
                manager._transition_phase = phase
                self.assertIsNone(
                    manager._stable_lifecycle_phase_locked(), msg=phase
                )
            manager._transition_phase = "idle"
            manager._lifecycle_state = "sleeping"
            self.assertEqual("idle", manager._stable_lifecycle_phase_locked())
            manager._transition_phase = "ready"
            manager._lifecycle_state = "active"
            self.assertEqual("ready", manager._stable_lifecycle_phase_locked())
            manager._transition_phase = "unknown"
            manager._lifecycle_state = "unknown"
            self.assertEqual("unknown", manager._stable_lifecycle_phase_locked())
        entries = (
            lambda: manager.startup_sleep_model(model().name),
            manager.finalize_startup,
            manager.adopt_startup_state,
            manager.reconcile_startup_state,
            lambda: manager.ensure_awake(model().name),
            lambda: manager.acquire(model().name),
            manager.sleep_active_model,
            manager.startup_state,
        )
        for entry in entries:
            with self.subTest(entry=getattr(entry, "__name__", repr(entry))):
                with self.assertRaisesRegex(WakeError, "projection is not active"):
                    entry()
        self.assertEqual([], manager.http.calls)

    def test_out_of_order_conflicting_and_timed_out_control_stays_fenced(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing")
            a = envelope(now, "active")
            self.write(path, a)
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            with self.assertRaisesRegex(WakeError, "no in-memory quiesce"):
                manager.projection_activate(
                    authority.authorize_control(
                        "activate", control_request(a, "activate")
                    )
                )
            bad_q = dict(q)
            bad_q["transition"] = "advance"
            bad_q["source_authority_sha256"] = "c" * 64
            bad_q["target_store_revision"] = 2
            bad_q["target_record_revision"] = 2
            bad_q["publication_id"] = semantic_id(bad_q)
            self.write(path, bad_q)
            bad_parsed = authority.authorize_control(
                "quiesce", control_request(bad_q, "quiesce")
            )
            with self.assertRaisesRegex(WakeError, "requires bootstrap"):
                manager.projection_quiesce(bad_parsed)
            timed = envelope(time.time(), "quiescing")
            timed["deadline_epoch"] = time.time() + 0.03
            timed["publication_id"] = semantic_id(timed)
            self.write(path, timed)
            parsed = authority.authorize_control(
                "quiesce", control_request(timed, "quiesce")
            )
            with manager._condition:
                manager._transition_phase = "starting"
            with self.assertRaisesRegex(WakeError, "deadline expired"):
                manager.projection_quiesce(parsed)
            self.assertEqual("quiescing", manager.projection_state)
            self.assertFalse(manager.projection_control_ready)
            self.assertEqual([], manager.http.calls)

    def test_raw_envelope_must_be_canonical_and_metadata_stable(self) -> None:
        now = time.time()
        value = envelope(now, "active")
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            variants = (
                b" " + canonical,
                json.dumps(value, separators=(",", ":")).encode(),
                canonical.replace(b'"requirement":"REQ', b'"requirement":"\\u0052EQ'),
            )
            for raw in variants:
                with self.subTest(raw=raw[:30]):
                    path.write_bytes(raw)
                    path.chmod(0o600)
                    with self.assertRaises(ThermalActionControlError):
                        authority.load_envelope()
            path.write_bytes(canonical)
            path.chmod(0o600)
            before = os.stat(path)
            fields = {
                name: getattr(before, name)
                for name in (
                    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
                    "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
                )
            }
            fields["st_mtime_ns"] += 1
            unstable = SimpleNamespace(**fields)
            with patch(
                "vllm_sleeper_proxy.thermal_control.os.fstat",
                side_effect=(before, unstable),
            ):
                with self.assertRaises(ThermalActionControlError):
                    authority.load_envelope()

    def test_semantic_ids_and_activation_times_are_exact(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing")
            a = envelope(now, "active")
            forged = dict(q)
            forged["publication_id"] = "f" * 64
            self.write(path, forged)
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            with self.assertRaises(ThermalActionControlError):
                authority.load_envelope()
            self.write(path, q)
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            manager.projection_quiesce(
                authority.authorize_control(
                    "quiesce", control_request(q, "quiesce")
                )
            )
            for field, delta in (("prepared_at_epoch", 0.25), ("deadline_epoch", -0.25)):
                with self.subTest(field=field):
                    changed = dict(a)
                    changed[field] = float(changed[field]) + delta
                    changed["publication_id"] = semantic_id(changed)
                    linked_q = dict(q)
                    linked_q["successor_publication_id"] = changed["publication_id"]
                    linked_q["publication_id"] = semantic_id(linked_q)
                    changed["predecessor_publication_id"] = linked_q["publication_id"]
                    changed["publication_id"] = semantic_id(changed)
                    # Re-ack a fresh manager against the Q that links this final.
                    other = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
                    other.configure_projection_control()
                    self.write(path, linked_q)
                    other.projection_quiesce(
                        authority.authorize_control(
                            "quiesce", control_request(linked_q, "quiesce")
                        )
                    )
                    self.write(path, changed)
                    parsed = authority.authorize_control(
                        "activate", control_request(changed, "activate")
                    )
                    with self.assertRaisesRegex(WakeError, "not linked"):
                        other.projection_activate(parsed)

    def test_expired_publication_replacement_restart_and_response_loss(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            q = envelope(now, "quiescing")
            original_final = envelope(now, "active")
            q["deadline_epoch"] = time.time() + 0.2
            q["publication_id"] = semantic_id(q)
            # Q's successor is not used before replacement, but semantic Q ID
            # intentionally does not depend on successor identity.
            self.write(path, q)
            authority = FileThermalActionAuthority(path, allow_legacy=False)
            manager = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            manager.configure_projection_control()
            manager.projection_quiesce(
                authority.authorize_control(
                    "quiesce", control_request(q, "quiesce")
                )
            )
            live_replacement, _ = replacement_pair(
                q, original_final, time.time()
            )
            self.write(path, live_replacement)
            live_value = authority.authorize_control(
                "quiesce", control_request(live_replacement, "quiesce")
            )
            with self.assertRaisesRegex(WakeError, "conflicting"):
                manager.projection_quiesce(live_value)
            threading.Event().wait(0.22)
            self.assertFalse(manager.projection_control_ready)
            with self.assertRaisesRegex(WakeError, "projection is not active"):
                manager.ensure_awake(model().name)
            replacement, final = replacement_pair(
                q, original_final, time.time()
            )
            self.assertNotEqual(
                q["target_authority_sha256"],
                replacement["target_authority_sha256"],
            )
            self.write(path, replacement)
            replacement_value = authority.authorize_control(
                "quiesce", control_request(replacement, "quiesce")
            )
            phase = manager.projection_quiesce(replacement_value)
            self.assertEqual("unknown", phase)
            # A restarted process has no ACK memory but accepts the exact
            # predecessor-linked replacement and remains fenced.
            restarted = ModelManager([model()], NoEngineHttp(), bootstrap_mode=True)
            restarted.configure_projection_control()
            self.assertEqual("unknown", restarted.projection_quiesce(replacement_value))
            self.assertFalse(restarted.projection_control_ready)
            self.write(path, final)
            final_value = authority.authorize_control(
                "activate", control_request(final, "activate")
            )
            self.assertEqual("unknown", manager.projection_activate(final_value))
            # Lost activation ACK: exact successor retry is idempotent.
            self.assertEqual("unknown", manager.projection_activate(final_value))
            # The predecessor can never be restored after successor adoption.
            with self.assertRaises(WakeError):
                manager.projection_quiesce(replacement_value)
            valid = authority.authorize(
                "hold", action_request(
                    final,
                    proxy_instance_id=manager.proxy_instance_id,
                    activation_token=manager.projection_activation_token or "",
                )
            )
            manager.require_projection_action(valid)
            guessed_from_q_instance = authority.authorize(
                "hold", action_request(
                    final,
                    proxy_instance_id=manager.proxy_instance_id,
                    activation_token=manager.proxy_instance_id,
                )
            )
            with self.assertRaises(WakeError):
                manager.require_projection_action(guessed_from_q_instance)
            wrong_instance = authority.authorize(
                "hold", action_request(
                    final,
                    proxy_instance_id="f" * 64,
                    activation_token=manager.projection_activation_token or "",
                )
            )
            with self.assertRaises(WakeError):
                manager.require_projection_action(wrong_instance)
            with self.assertRaises(WakeError):
                restarted.require_projection_action(valid)
            self.assertEqual([], manager.http.calls)

    def test_control_enabled_build_binds_without_startup_reconcile(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            self.write(path, envelope(now, "quiescing"))
            manager = ModelManager([model()], NoEngineHttp())
            manager.reconcile_startup_state = lambda: self.fail("must not reconcile")
            server = build_server(
                "127.0.0.1", 0, manager,
                thermal_action_control_enabled=True,
                thermal_action_authority=FileThermalActionAuthority(path, allow_legacy=False),
            )
            try:
                self.assertEqual("unbound", manager.projection_state)
                self.assertFalse(manager.projection_control_ready)
                self.assertEqual([], manager.http.calls)
            finally:
                server.server_close()


if __name__ == "__main__":
    unittest.main()
