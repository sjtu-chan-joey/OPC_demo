"""Battery simulation core with an optional PyBaMM entry point.

The built-in model is a calibrated equivalent aging model. It is not meant to
replace PyBaMM physics, but it is fast, deterministic, and exposes the same
decision variables required for LLM policy learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Iterable

import pandas as pd

from .protocol_space import ProtocolAction, default_actions


@dataclass
class BatteryState:
    episode_id: int
    step_id: int
    time_min: float
    soc: float
    soh: float
    capacity_ah: float
    resistance_mohm: float
    temperature_c: float
    ambient_c: float
    voltage_v: float
    cumulative_ah: float
    cumulative_energy_wh: float
    last_mode: str = "rest"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scenario:
    episode_id: int
    ambient_c: float
    initial_soc: float
    initial_soh: float
    initial_resistance_mohm: float
    nominal_capacity_ah: float = 3.0
    objective: str = "balance_capacity_speed_energy_and_soc_trajectory"


@dataclass(frozen=True)
class MultiObjectiveWeights:
    """Weights for scalarizing the protocol objectives.

    The scalar score is only used to label oracle actions and rank candidates.
    Validation reports each objective separately.
    """

    capacity_retention: float = 90.0
    soc_window: float = 0.75
    energy_throughput: float = 0.045
    cycle_progress: float = 3.5
    time_efficiency: float = 0.004
    temperature_safety: float = 0.10
    voltage_safety: float = 3.0
    high_soc_penalty: float = 0.45
    rest_penalty: float = 0.14


def ocv_from_soc(soc: float) -> float:
    """Approximate graphite/NMC open-circuit voltage curve."""

    soc = min(1.0, max(0.0, soc))
    return 3.0 + 1.18 * soc + 0.08 * math.sin(math.pi * (soc - 0.1))


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def initial_state(scenario: Scenario) -> BatteryState:
    soc = _clip(scenario.initial_soc, 0.05, 0.98)
    capacity = scenario.nominal_capacity_ah * scenario.initial_soh
    return BatteryState(
        episode_id=scenario.episode_id,
        step_id=0,
        time_min=0.0,
        soc=soc,
        soh=scenario.initial_soh,
        capacity_ah=capacity,
        resistance_mohm=scenario.initial_resistance_mohm,
        temperature_c=scenario.ambient_c,
        ambient_c=scenario.ambient_c,
        voltage_v=ocv_from_soc(soc),
        cumulative_ah=0.0,
        cumulative_energy_wh=0.0,
    )


def simulate_action(
    state: BatteryState,
    action: ProtocolAction,
    nominal_capacity_ah: float = 3.0,
    integration_min: float = 1.0,
) -> tuple[BatteryState, list[dict]]:
    """Advance the state through one protocol segment."""

    next_state = BatteryState(**state.to_dict())
    rows: list[dict] = []
    segment_minutes = max(1, int(action.duration_min))
    remaining = float(segment_minutes)

    while remaining > 0:
        dt_min = min(integration_min, remaining)
        remaining -= dt_min
        sign = 1.0 if action.mode == "charge" else -1.0 if action.mode == "discharge" else 0.0
        current_a = sign * action.c_rate * nominal_capacity_ah
        dt_h = dt_min / 60.0
        delta_ah = current_a * dt_h

        if action.mode == "charge" and action.target_soc is not None and next_state.soc >= action.target_soc:
            current_a = 0.0
            delta_ah = 0.0
        if action.mode == "discharge" and action.target_soc is not None and next_state.soc <= action.target_soc:
            current_a = 0.0
            delta_ah = 0.0

        next_state.soc = _clip(next_state.soc + delta_ah / max(next_state.capacity_ah, 0.2), 0.02, 0.99)
        ohmic_v = current_a * next_state.resistance_mohm / 1000.0
        next_state.voltage_v = _clip(ocv_from_soc(next_state.soc) + ohmic_v, 2.5, 4.35)

        heat_w = (current_a**2) * next_state.resistance_mohm / 1000.0
        temp_relax = (next_state.temperature_c - next_state.ambient_c) * 0.018 * dt_min
        next_state.temperature_c += heat_w * 0.075 * dt_min - temp_relax

        throughput_ah = abs(delta_ah)
        temp_stress = math.exp(max(0.0, next_state.temperature_c - 25.0) / 17.0)
        cold_charge = 1.0 + (1.8 if action.mode == "charge" and next_state.temperature_c < 5.0 and action.c_rate >= 0.75 else 0.0)
        high_soc_stress = 1.0 + 2.2 * max(0.0, next_state.soc - 0.82) ** 2
        low_soc_stress = 1.0 + 0.6 * max(0.0, 0.15 - next_state.soc)
        rate_stress = 1.0 + max(0.0, action.c_rate - 0.5) ** 1.7
        calendar_fade = 1.2e-7 * dt_min * (1.0 + max(0.0, next_state.temperature_c - 25.0) / 20.0)
        cycle_fade = 4.0e-5 * throughput_ah * temp_stress * cold_charge * high_soc_stress * low_soc_stress * rate_stress
        soh_loss = calendar_fade + cycle_fade
        next_state.soh = _clip(next_state.soh - soh_loss, 0.5, 1.0)
        next_state.capacity_ah = nominal_capacity_ah * next_state.soh
        next_state.resistance_mohm += 0.9 * soh_loss * 1000.0 + 0.0009 * throughput_ah
        next_state.cumulative_ah += throughput_ah
        next_state.cumulative_energy_wh += abs(delta_ah) * next_state.voltage_v
        next_state.time_min += dt_min
        next_state.last_mode = action.mode

        row = next_state.to_dict()
        row.update(
            {
                "action_id": action.action_id,
                "mode": action.mode,
                "c_rate": action.c_rate,
                "duration_min": action.duration_min,
                "target_soc": action.target_soc,
                "voltage_limit_v": action.voltage_limit_v,
                "current_a": current_a,
            }
        )
        rows.append(row)

    next_state.step_id += 1
    return next_state, rows


def equivalent_cycles(state: BatteryState, nominal_capacity_ah: float = 3.0) -> float:
    """Full equivalent cycles from cumulative charge/discharge throughput."""

    return state.cumulative_ah / max(2.0 * nominal_capacity_ah, 1.0e-9)


def transition_objectives(
    before: BatteryState,
    after: BatteryState,
    action: ProtocolAction,
    nominal_capacity_ah: float = 3.0,
) -> dict[str, float]:
    """Return objective components for one decision segment."""

    soh_loss = before.soh - after.soh
    energy_delta_wh = after.cumulative_energy_wh - before.cumulative_energy_wh
    ah_delta = after.cumulative_ah - before.cumulative_ah
    cycle_delta = ah_delta / max(2.0 * nominal_capacity_ah, 1.0e-9)
    duration_h = max(action.duration_min / 60.0, 1.0e-9)
    energy_rate_w = energy_delta_wh / duration_h
    soc_mid = 0.55
    soc_window_error = abs(after.soc - soc_mid)
    high_soc_exposure = max(0.0, after.soc - 0.86) * max(action.duration_min, 1) / 60.0
    temperature_risk = max(0.0, after.temperature_c - 40.0)
    voltage_risk = max(0.0, after.voltage_v - 4.2) + max(0.0, 3.0 - after.voltage_v)
    return {
        "soh_loss": soh_loss,
        "capacity_loss_ah": before.capacity_ah - after.capacity_ah,
        "soc_delta": after.soc - before.soc,
        "soc_window_error": soc_window_error,
        "energy_delta_wh": energy_delta_wh,
        "ah_delta": ah_delta,
        "cycle_delta": cycle_delta,
        "energy_rate_w": energy_rate_w,
        "temperature_risk": temperature_risk,
        "voltage_risk": voltage_risk,
        "high_soc_exposure": high_soc_exposure,
        "time_min": float(action.duration_min),
        "is_rest": 1.0 if action.mode == "rest" else 0.0,
    }


def score_transition(
    before: BatteryState,
    after: BatteryState,
    action: ProtocolAction,
    weights: MultiObjectiveWeights | None = None,
    nominal_capacity_ah: float = 3.0,
) -> float:
    """Higher is better; balances life, speed, energy, SOC trajectory, and safety."""

    w = weights or MultiObjectiveWeights()
    obj = transition_objectives(before, after, action, nominal_capacity_ah)
    cooling_rest_bonus = 0.035 if action.mode == "rest" and before.temperature_c > 37.0 else 0.0
    return (
        -w.capacity_retention * obj["soh_loss"]
        -w.soc_window * obj["soc_window_error"]
        +w.energy_throughput * obj["energy_delta_wh"]
        +w.cycle_progress * obj["cycle_delta"]
        +w.time_efficiency * obj["energy_rate_w"]
        -w.temperature_safety * obj["temperature_risk"]
        -w.voltage_safety * obj["voltage_risk"]
        -w.high_soc_penalty * obj["high_soc_exposure"]
        -w.rest_penalty * obj["is_rest"]
        +cooling_rest_bonus
    )


def feasible_actions(state: BatteryState, actions: Iterable[ProtocolAction]) -> list[ProtocolAction]:
    candidates: list[ProtocolAction] = []
    for action in actions:
        if action.mode == "charge" and state.soc >= 0.96:
            continue
        if action.mode == "discharge" and state.soc <= 0.18:
            continue
        if action.mode == "rest":
            candidates.append(action)
        elif action.mode == "charge" and action.target_soc is not None and action.target_soc > state.soc:
            candidates.append(action)
        elif action.mode == "discharge" and action.target_soc is not None and action.target_soc < state.soc:
            candidates.append(action)
    return candidates


def choose_oracle_action(state: BatteryState, actions: Iterable[ProtocolAction], nominal_capacity_ah: float) -> tuple[ProtocolAction, float]:
    best_action: ProtocolAction | None = None
    best_score = float("-inf")
    for action in feasible_actions(state, actions):
        after, _ = simulate_action(state, action, nominal_capacity_ah=nominal_capacity_ah)
        score = score_transition(state, after, action, nominal_capacity_ah=nominal_capacity_ah)
        if score > best_score:
            best_action = action
            best_score = score
    if best_action is None:
        rest = ProtocolAction("rst_010", "rest", 0.0, 10, None, None, 0.0)
        return rest, 0.0
    return best_action, best_score


def make_scenarios(count: int, seed: int = 7) -> list[Scenario]:
    rng = random.Random(seed)
    ambients = [-5.0, 5.0, 15.0, 25.0, 35.0, 45.0]
    scenarios: list[Scenario] = []
    for episode_id in range(count):
        initial_soh = rng.uniform(0.86, 1.0)
        scenarios.append(
            Scenario(
                episode_id=episode_id,
                ambient_c=rng.choice(ambients) + rng.uniform(-2.0, 2.0),
                initial_soc=rng.uniform(0.12, 0.9),
                initial_soh=initial_soh,
                initial_resistance_mohm=rng.uniform(35.0, 70.0) + (1.0 - initial_soh) * 55.0,
            )
        )
    return scenarios


def generate_dataset(
    episodes: int,
    steps_per_episode: int,
    seed: int = 7,
    nominal_capacity_ah: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate transition labels and time-series rows."""

    actions = default_actions(nominal_capacity_ah)
    scenarios = make_scenarios(episodes, seed)
    transitions: list[dict] = []
    time_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for scenario in scenarios:
        state = initial_state(scenario)
        for _ in range(steps_per_episode):
            candidates = feasible_actions(state, actions)
            ranked: list[tuple[float, ProtocolAction, BatteryState]] = []
            for action in candidates:
                after_candidate, _ = simulate_action(state, action, nominal_capacity_ah)
                score = score_transition(state, after_candidate, action, nominal_capacity_ah=nominal_capacity_ah)
                objectives = transition_objectives(state, after_candidate, action, nominal_capacity_ah)
                ranked.append((score, action, after_candidate))
                candidate_row = {
                        "episode_id": state.episode_id,
                        "step_id": state.step_id,
                        "candidate_action_id": action.action_id,
                        "candidate_mode": action.mode,
                        "candidate_c_rate": action.c_rate,
                        "candidate_duration_min": action.duration_min,
                        "candidate_target_soc": action.target_soc,
                        "candidate_score": score,
                        "predicted_soh_after": after_candidate.soh,
                        "predicted_capacity_ah_after": after_candidate.capacity_ah,
                        "predicted_soc_after": after_candidate.soc,
                        "predicted_temperature_c_after": after_candidate.temperature_c,
                        "predicted_energy_wh_after": after_candidate.cumulative_energy_wh,
                    }
                candidate_row.update({f"candidate_{key}": value for key, value in objectives.items()})
                candidate_rows.append(candidate_row)
            ranked.sort(key=lambda item: item[0], reverse=True)
            score, action, after = ranked[0]
            before = state
            state, rows = simulate_action(before, action, nominal_capacity_ah)
            objectives = transition_objectives(before, state, action, nominal_capacity_ah)
            time_rows.extend(rows)
            transition = before.to_dict()
            transition.update(
                {
                    "selected_action_id": action.action_id,
                    "selected_mode": action.mode,
                    "selected_c_rate": action.c_rate,
                    "selected_duration_min": action.duration_min,
                    "selected_target_soc": action.target_soc,
                    "selected_voltage_limit_v": action.voltage_limit_v,
                    "reward_score": score,
                    "equivalent_cycles": equivalent_cycles(state, nominal_capacity_ah),
                    "next_soc": state.soc,
                    "next_soh": state.soh,
                    "next_capacity_ah": state.capacity_ah,
                    "next_temperature_c": state.temperature_c,
                    "next_resistance_mohm": state.resistance_mohm,
                    "next_cumulative_energy_wh": state.cumulative_energy_wh,
                }
            )
            transition.update(objectives)
            transitions.append(transition)
            if state.soh <= 0.80:
                break

    return pd.DataFrame(transitions), pd.DataFrame(time_rows), pd.DataFrame(candidate_rows)


def pybamm_available() -> bool:
    try:
        import pybamm  # noqa: F401
    except Exception:
        return False
    return True
