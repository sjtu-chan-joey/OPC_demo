"""Local dashboard server for real-time closed-loop battery control demos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import uuid

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battery_llm_control.policy import FixedPolicy, LLMJsonPolicy, NearestNeighborPolicy, OraclePolicy
from battery_llm_control.protocol_space import ProtocolAction
from battery_llm_control.sim_core import BatteryState, Scenario, initial_state, simulate_action


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "dashboard"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TRANSITIONS = ROOT / "data/simulated/transitions.csv"
DEFAULT_MODEL_PATH = ROOT / "adapters/qwen3_battery_lora"
DEFAULT_DECISION_CYCLE_INTERVAL = 0.10
MAX_NO_PROGRESS_DECISIONS = 20
SESSION_TTL_SECONDS = 60 * 60


@dataclass
class DashboardConfig:
    host: str
    port: int
    transitions_csv: Path
    model_path: str | None


@dataclass
class PolicyRuntime:
    key: str
    label: str
    policy: Any
    state: BatteryState
    rows: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    decision_step: int = 0
    decision_target_cycle: float = 0.0
    current_action: ProtocolAction | None = None
    done: bool = False
    no_progress_decisions: int = 0


@dataclass
class DashboardSession:
    session_id: str
    scenario: Scenario
    target_cycles: float
    decision_cycle_interval: float
    policies: list[PolicyRuntime]
    llm_status: str
    has_real_llm: bool
    created_at: float
    updated_at: float


SESSIONS: dict[str, DashboardSession] = {}
SESSIONS_LOCK = threading.Lock()


class ConservativePolicy:
    """Battery-life-first hand-crafted baseline."""

    def select_action(self, state: BatteryState) -> ProtocolAction:
        if state.temperature_c > 38.0:
            return ProtocolAction("safe_rest_hot", "rest", 0.0, 20, None, None, 0.0)
        if state.soc < 0.76 and state.last_mode != "discharge":
            rate = 0.33 if state.temperature_c < 10.0 or state.temperature_c > 34.0 else 0.5
            return ProtocolAction("safe_charge", "charge", rate, 30, 0.76, 4.1, rate * 3.0)
        if state.soc > 0.36:
            return ProtocolAction("safe_discharge", "discharge", 0.5, 30, 0.36, 3.0, 1.5)
        return ProtocolAction("safe_rest_mid", "rest", 0.0, 10, None, None, 0.0)


class AdaptiveLLMSurrogatePolicy:
    """Fast local LLM-shaped policy used when the fine-tuned model is unavailable."""

    def select_action(self, state: BatteryState) -> ProtocolAction:
        hot = state.temperature_c > 40.0 or state.ambient_c >= 43.0
        cold = state.temperature_c < 6.0
        aged = state.soh < 0.9 or state.resistance_mohm > 72.0
        if hot:
            return ProtocolAction("llm_dynamic_rest_hot", "rest", 0.0, 20, None, None, 0.0)

        rate = 0.33 if cold or aged else 0.75
        if state.temperature_c > 34.0:
            rate = min(rate, 0.5)

        if state.soc < 0.80 and state.last_mode != "discharge":
            return ProtocolAction("llm_dynamic_charge_low", "charge", rate, 30, 0.80, 4.1, rate * 3.0)
        if state.soc > 0.28:
            discharge_rate = 0.5 if aged or state.temperature_c > 34.0 else 1.0
            return ProtocolAction(
                "llm_dynamic_discharge_high", "discharge", discharge_rate, 30, 0.28, 3.0, discharge_rate * 3.0
            )
        if state.last_mode == "charge" and state.soc < 0.78:
            return ProtocolAction("llm_dynamic_charge_follow", "charge", rate, 20, 0.80, 4.1, rate * 3.0)
        if state.last_mode == "discharge" and state.soc > 0.32:
            return ProtocolAction("llm_dynamic_discharge_follow", "discharge", 0.5, 20, 0.28, 3.0, 1.5)
        return ProtocolAction("llm_dynamic_rest_balance", "rest", 0.0, 10, None, None, 0.0)


def _float_param(params: dict[str, list[str]], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(params.get(key, [default])[0])
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def _equivalent_cycles(state: BatteryState, nominal_capacity_ah: float) -> float:
    return state.cumulative_ah / max(2.0 * nominal_capacity_ah, 1.0e-9)


def _limit_action_to_cycle_target(
    state: BatteryState,
    action: ProtocolAction,
    target_cycle: float,
    nominal_capacity_ah: float,
) -> ProtocolAction:
    if action.mode == "rest" or action.c_rate <= 0.0:
        return replace(action, duration_min=1)
    remaining_ah = target_cycle * 2.0 * nominal_capacity_ah - state.cumulative_ah
    if remaining_ah <= 0.0:
        return replace(action, duration_min=1)
    ah_per_min = action.c_rate * nominal_capacity_ah / 60.0
    max_duration = max(1, int(remaining_ah / ah_per_min))
    return replace(action, duration_min=min(1, max_duration))


def _policy_reason(policy_key: str, state: BatteryState, action: ProtocolAction) -> str:
    if policy_key == "llm":
        if action.mode == "rest":
            return "LLM detected temperature, SOC, or aging risk and selected rest."
        return f"LLM reads SOC={state.soc:.2f}, SOH={state.soh:.3f}, T={state.temperature_c:.1f}C and selects {action.mode}."
    if policy_key == "oracle":
        return "Oracle evaluates candidate protocols in the simulator and picks the best immediate score."
    if policy_key == "nearest":
        return "Nearest-neighbor policy matches the current state to historical simulated transitions."
    if policy_key == "conservative":
        return "Conservative policy limits high SOC, high C-rate, and high-temperature operation."
    return "Fixed baseline follows a simple 1C high-SOC cycling rule."


def _initial_row(key: str, label: str, state: BatteryState) -> dict[str, Any]:
    row = state.to_dict()
    row.update(
        {
            "policy_key": key,
            "policy_label": label,
            "decision_step": 0,
            "decision_start_cycle": 0.0,
            "decision_target_cycle": 0.0,
            "action_id": "initial",
            "mode": "rest",
            "c_rate": 0.0,
            "duration_min": 0,
            "target_soc": None,
            "voltage_limit_v": None,
            "current_a": 0.0,
            "soh_loss": 0.0,
            "equivalent_cycles": 0.0,
            "event": "initial",
        }
    )
    return row


def _build_llm_policy(config: DashboardConfig, scenario: Scenario) -> tuple[Any, str, bool]:
    if not config.model_path:
        return AdaptiveLLMSurrogatePolicy(), "Fast LLM surrogate", False
    try:
        return LLMJsonPolicy(config.model_path, scenario.nominal_capacity_ah), "Local GPU LLM", True
    except Exception as exc:
        return AdaptiveLLMSurrogatePolicy(), f"Local LLM unavailable, using fast surrogate: {exc}", False


def _create_session(config: DashboardConfig, params: dict[str, list[str]]) -> DashboardSession:
    target_cycles = _float_param(params, "target_cycles", 1.0, 0.2, 80.0)
    decision_cycle_interval = _float_param(
        params,
        "decision_cycle_interval",
        DEFAULT_DECISION_CYCLE_INTERVAL,
        0.01,
        10.0,
    )
    scenario = Scenario(
        episode_id=20260629,
        ambient_c=_float_param(params, "ambient_c", 35.0, -10.0, 50.0),
        initial_soc=_float_param(params, "initial_soc", 0.35, 0.08, 0.92),
        initial_soh=_float_param(params, "initial_soh", 0.94, 0.82, 1.0),
        initial_resistance_mohm=_float_param(params, "resistance_mohm", 58.0, 25.0, 110.0),
        nominal_capacity_ah=_float_param(params, "nominal_capacity_ah", 3.0, 1.0, 8.0),
    )

    llm_policy, llm_status, has_real_llm = _build_llm_policy(config, scenario)
    policy_specs: list[tuple[str, str, Any]] = [
        ("llm", "LLM dynamic policy", llm_policy),
        ("oracle", "Oracle upper-bound policy", OraclePolicy(scenario.nominal_capacity_ah)),
        ("conservative", "Conservative life policy", ConservativePolicy()),
        ("fixed", "Fixed 1C baseline", FixedPolicy()),
    ]
    if config.transitions_csv.exists():
        policy_specs.insert(2, ("nearest", "Historical nearest-neighbor policy", NearestNeighborPolicy(str(config.transitions_csv))))

    runtimes: list[PolicyRuntime] = []
    for key, label, policy in policy_specs:
        state = initial_state(scenario)
        runtimes.append(PolicyRuntime(key=key, label=label, policy=policy, state=state, rows=[_initial_row(key, label, state)], decisions=[]))

    now = time.time()
    return DashboardSession(
        session_id=uuid.uuid4().hex,
        scenario=scenario,
        target_cycles=target_cycles,
        decision_cycle_interval=decision_cycle_interval,
        policies=runtimes,
        llm_status=llm_status,
        has_real_llm=has_real_llm,
        created_at=now,
        updated_at=now,
    )


def _maybe_select_action(session: DashboardSession, runtime: PolicyRuntime) -> dict[str, Any] | None:
    cycle = _equivalent_cycles(runtime.state, session.scenario.nominal_capacity_ah)
    needs_decision = runtime.current_action is None or cycle >= runtime.decision_target_cycle
    if not needs_decision:
        return None

    runtime.decision_step += 1
    decision_start_cycle = cycle
    runtime.decision_target_cycle = min(cycle + session.decision_cycle_interval, session.target_cycles)
    runtime.current_action = runtime.policy.select_action(runtime.state)
    decision = {
        "decision_step": runtime.decision_step,
        "time_min": runtime.state.time_min,
        "soc": runtime.state.soc,
        "soh": runtime.state.soh,
        "temperature_c": runtime.state.temperature_c,
        "decision_start_cycle": decision_start_cycle,
        "decision_target_cycle": runtime.decision_target_cycle,
        "action": runtime.current_action.to_dict(),
        "reason": _policy_reason(runtime.key, runtime.state, runtime.current_action),
    }
    runtime.decisions.append(decision)
    return decision


def _step_policy(session: DashboardSession, runtime: PolicyRuntime) -> dict[str, Any]:
    if runtime.done:
        return {"key": runtime.key, "done": True, "new_rows": [], "new_decision": None}

    current_cycle = _equivalent_cycles(runtime.state, session.scenario.nominal_capacity_ah)
    if current_cycle >= session.target_cycles or runtime.state.soh <= 0.80:
        runtime.done = True
        return {"key": runtime.key, "done": True, "new_rows": [], "new_decision": None}

    decision = _maybe_select_action(session, runtime)
    action = runtime.current_action
    if action is None:
        runtime.done = True
        return {"key": runtime.key, "done": True, "new_rows": [], "new_decision": decision}

    before = runtime.state
    before_cycle = _equivalent_cycles(before, session.scenario.nominal_capacity_ah)
    one_min_action = _limit_action_to_cycle_target(
        before,
        action,
        min(runtime.decision_target_cycle, session.target_cycles),
        session.scenario.nominal_capacity_ah,
    )
    after, minute_rows = simulate_action(before, one_min_action, session.scenario.nominal_capacity_ah)
    runtime.state = after

    new_rows = []
    for row in minute_rows:
        row.update(
            {
                "policy_key": runtime.key,
                "policy_label": runtime.label,
                "decision_step": runtime.decision_step,
                "decision_start_cycle": runtime.decisions[-1]["decision_start_cycle"] if runtime.decisions else 0.0,
                "decision_target_cycle": runtime.decision_target_cycle,
                "soh_loss": before.soh - row["soh"],
                "equivalent_cycles": row["cumulative_ah"] / max(2.0 * session.scenario.nominal_capacity_ah, 1.0e-9),
                "event": "simulate",
            }
        )
        runtime.rows.append(row)
        new_rows.append(row)

    after_cycle = _equivalent_cycles(runtime.state, session.scenario.nominal_capacity_ah)
    progressed = after_cycle > before_cycle + 1.0e-9
    if progressed:
        runtime.no_progress_decisions = 0
    elif action.mode == "rest" or action.c_rate <= 0.0:
        runtime.current_action = None
        runtime.no_progress_decisions += 1
    else:
        runtime.current_action = None

    if after_cycle >= session.target_cycles or runtime.state.soh <= 0.80 or runtime.no_progress_decisions >= MAX_NO_PROGRESS_DECISIONS:
        runtime.done = True

    return {"key": runtime.key, "done": runtime.done, "new_rows": new_rows, "new_decision": decision}


def _summary(runtime: PolicyRuntime) -> dict[str, Any]:
    final = runtime.rows[-1]
    return {
        "final_soc": final["soc"],
        "final_soh": final["soh"],
        "final_capacity_ah": final["capacity_ah"],
        "max_temperature_c": max(row["temperature_c"] for row in runtime.rows),
        "total_energy_wh": final["cumulative_energy_wh"],
        "total_ah": final["cumulative_ah"],
        "equivalent_cycles": final["equivalent_cycles"],
        "time_min": final["time_min"],
        "decision_steps": len(runtime.decisions),
        "done": runtime.done,
    }


def _full_session_payload(session: DashboardSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "scenario": session.scenario.__dict__,
        "target_cycles": session.target_cycles,
        "decision_cycle_interval": session.decision_cycle_interval,
        "has_real_llm": session.has_real_llm,
        "llm_status": session.llm_status,
        "policies": [
            {
                "key": runtime.key,
                "label": runtime.label,
                "rows": runtime.rows,
                "decisions": runtime.decisions,
                "summary": _summary(runtime),
            }
            for runtime in session.policies
        ],
        "done": all(runtime.done for runtime in session.policies),
    }


def _step_session(session: DashboardSession) -> dict[str, Any]:
    updates = [_step_policy(session, runtime) for runtime in session.policies]
    session.updated_at = time.time()
    return {
        "session_id": session.session_id,
        "target_cycles": session.target_cycles,
        "decision_cycle_interval": session.decision_cycle_interval,
        "has_real_llm": session.has_real_llm,
        "llm_status": session.llm_status,
        "updates": updates,
        "policies": [
            {
                "key": runtime.key,
                "label": runtime.label,
                "summary": _summary(runtime),
            }
            for runtime in session.policies
        ],
        "done": all(runtime.done for runtime in session.policies),
    }


def _cleanup_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [session_id for session_id, session in SESSIONS.items() if session.updated_at < cutoff]
    for session_id in expired:
        del SESSIONS[session_id]


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/api/start":
            self._handle_start(params)
            return
        if parsed.path == "/api/step":
            self._handle_step(params)
            return
        if parsed.path == "/api/simulate":
            self._handle_compat_simulate(params)
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        static_path = STATIC_DIR / parsed.path.lstrip("/")
        content_types = {
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".html": "text/html; charset=utf-8",
        }
        if static_path.is_file() and static_path.resolve().is_relative_to(STATIC_DIR.resolve()):
            self._send_file(static_path, content_types.get(static_path.suffix, "application/octet-stream"))
            return
        self.send_error(404, "Not found")

    def _handle_start(self, params: dict[str, list[str]]) -> None:
        try:
            session = _create_session(self.config, params)
            with SESSIONS_LOCK:
                _cleanup_sessions()
                SESSIONS[session.session_id] = session
            self._send_json(_full_session_payload(session))
        except Exception as exc:
            self._send_error_json(exc)

    def _handle_step(self, params: dict[str, list[str]]) -> None:
        session_id = params.get("session_id", [""])[0]
        with SESSIONS_LOCK:
            session = SESSIONS.get(session_id)
            if session is None:
                self._send_json({"error": "session not found or expired"}, status=404)
                return
            try:
                payload = _step_session(session)
            except Exception as exc:
                self._send_error_json(exc)
                return
        self._send_json(payload)

    def _handle_compat_simulate(self, params: dict[str, list[str]]) -> None:
        try:
            session = _create_session(self.config, params)
            while not all(runtime.done for runtime in session.policies):
                _step_session(session)
            self._send_json(_full_session_payload(session))
        except Exception as exc:
            self._send_error_json(exc)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, exc: Exception) -> None:
        self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the battery LLM control dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Optional local fine-tuned model path.")
    args = parser.parse_args()

    config = DashboardConfig(args.host, args.port, args.transitions, args.model_path)
    DashboardHandler.config = config
    server = ThreadingHTTPServer((config.host, config.port), DashboardHandler)
    print(f"Dashboard running at http://{config.host}:{config.port}")
    print("Open the URL in a browser, set parameters, then press Start.")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
