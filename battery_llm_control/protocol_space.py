"""Discrete protocol action space for battery charging control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class ProtocolAction:
    """One controllable charge/discharge/rest segment.

    Values are deliberately quantized so they can be emitted by an LLM as JSON.
    A physical controller can map the same schema to power-supply commands.
    """

    action_id: str
    mode: str
    c_rate: float
    duration_min: int
    target_soc: float | None
    voltage_limit_v: float | None
    current_limit_a: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def default_actions(nominal_capacity_ah: float = 3.0) -> list[ProtocolAction]:
    """Return a broad but bounded action space for optimization.

    The space includes multiple C-rates, segment durations, and target SOCs.
    It intentionally keeps voltages inside common Li-ion operating limits.
    """

    actions: list[ProtocolAction] = []
    charge_rates = [0.2, 0.33, 0.5, 0.75, 1.0, 1.25]
    discharge_rates = [0.2, 0.5, 1.0, 1.5]
    durations = [10, 20, 30, 45, 60]
    charge_targets = [0.55, 0.7, 0.8, 0.9, 0.98]
    discharge_targets = [0.15, 0.2, 0.3, 0.4, 0.5]

    for idx, (c_rate, duration_min, target_soc) in enumerate(
        product(charge_rates, durations, charge_targets), start=1
    ):
        actions.append(
            ProtocolAction(
                action_id=f"chg_{idx:03d}",
                mode="charge",
                c_rate=c_rate,
                duration_min=duration_min,
                target_soc=target_soc,
                voltage_limit_v=4.2 if target_soc > 0.9 else 4.1,
                current_limit_a=round(c_rate * nominal_capacity_ah, 3),
            )
        )

    for idx, (c_rate, duration_min, target_soc) in enumerate(
        product(discharge_rates, durations, discharge_targets), start=1
    ):
        actions.append(
            ProtocolAction(
                action_id=f"dis_{idx:03d}",
                mode="discharge",
                c_rate=c_rate,
                duration_min=duration_min,
                target_soc=target_soc,
                voltage_limit_v=3.0,
                current_limit_a=round(c_rate * nominal_capacity_ah, 3),
            )
        )

    for duration_min in [5, 10, 20, 30, 60]:
        actions.append(
            ProtocolAction(
                action_id=f"rst_{duration_min:03d}",
                mode="rest",
                c_rate=0.0,
                duration_min=duration_min,
                target_soc=None,
                voltage_limit_v=None,
                current_limit_a=0.0,
            )
        )

    return actions


def actions_by_mode(actions: Iterable[ProtocolAction], mode: str) -> list[ProtocolAction]:
    return [action for action in actions if action.mode == mode]
