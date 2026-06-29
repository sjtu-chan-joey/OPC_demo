"""Control policies for closed-loop validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Protocol

import pandas as pd

from .protocol_space import ProtocolAction, default_actions
from .sim_core import BatteryState, choose_oracle_action


ROOT = Path(__file__).resolve().parents[1]


class BatteryPolicy(Protocol):
    def select_action(self, state: BatteryState) -> ProtocolAction:
        ...


@dataclass
class FixedPolicy:
    mode: str = "charge"
    c_rate: float = 1.0
    duration_min: int = 60
    target_soc: float = 0.98

    def select_action(self, state: BatteryState) -> ProtocolAction:
        if state.soc >= 0.95:
            return ProtocolAction("fixed_discharge", "discharge", 1.0, 60, 0.2, 3.0, 3.0)
        if state.temperature_c > 40.0:
            return ProtocolAction("fixed_rest", "rest", 0.0, 20, None, None, 0.0)
        return ProtocolAction("fixed_charge", self.mode, self.c_rate, self.duration_min, self.target_soc, 4.2, 3.0)


@dataclass
class OraclePolicy:
    nominal_capacity_ah: float = 3.0

    def select_action(self, state: BatteryState) -> ProtocolAction:
        action, _ = choose_oracle_action(state, default_actions(self.nominal_capacity_ah), self.nominal_capacity_ah)
        return action


@dataclass
class NearestNeighborPolicy:
    transitions_csv: str

    def __post_init__(self) -> None:
        self.transitions = pd.read_csv(self.transitions_csv)

    def select_action(self, state: BatteryState) -> ProtocolAction:
        df = self.transitions
        distance = (
            (df["soc"] - state.soc).abs() * 3.0
            + (df["soh"] - state.soh).abs() * 5.0
            + (df["temperature_c"] - state.temperature_c).abs() / 30.0
            + (df["ambient_c"] - state.ambient_c).abs() / 40.0
            + (df["resistance_mohm"] - state.resistance_mohm).abs() / 80.0
        )
        row = df.loc[distance.idxmin()]
        return ProtocolAction(
            action_id=str(row["selected_action_id"]),
            mode=str(row["selected_mode"]),
            c_rate=float(row["selected_c_rate"]),
            duration_min=int(row["selected_duration_min"]),
            target_soc=None if pd.isna(row["selected_target_soc"]) else float(row["selected_target_soc"]),
            voltage_limit_v=None if pd.isna(row["selected_voltage_limit_v"]) else float(row["selected_voltage_limit_v"]),
            current_limit_a=float(row["selected_c_rate"]) * 3.0,
        )


class LLMJsonPolicy:
    """Local Hugging Face JSON policy.

    The LLM path is intentionally GPU-only. If CUDA is not available, failing
    early is safer than silently pushing a multi-billion-parameter model onto
    the CPU and making validation appear hung.
    """

    def __init__(self, model_path: str, nominal_capacity_ah: float = 3.0):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LLMJsonPolicy requires CUDA. Install a CUDA-enabled PyTorch build and GPU driver.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.nominal_capacity_ah = nominal_capacity_ah
        self.device = torch.device("cuda")
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        path = self._resolve_local_model_path(model_path)
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel

            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model = self._resolve_local_model_path(
                config.get("base_model_name_or_path", "./models/Qwen3-4B-Instruct-2507")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=self.torch_dtype,
            )
            self.model = PeftModel.from_pretrained(base, path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=self.torch_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_local_model_path(model_path: str) -> Path:
        path = Path(model_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.append(ROOT / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Local model path was not found. "
            f"Checked: {checked}. "
            "Pass an existing local directory with --model-path, or run non-LLM validation with "
            "--policies fixed,balanced,conservative,oracle,nearest."
        )

    def select_action(self, state: BatteryState) -> ProtocolAction:
        import torch

        prompt = (
            "你是电池充放电优化控制器。只输出一个 JSON 对象，不要解释。\n"
            f"当前状态: {json.dumps(state.to_dict(), ensure_ascii=False)}\n"
            "字段: mode,c_rate,duration_min,target_soc,voltage_limit_v\n"
            "mode 只能是 charge、discharge 或 rest。"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)

        generated_ids = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            return OraclePolicy(self.nominal_capacity_ah).select_action(state)

        try:
            payload = json.loads(match.group(0))
            return self._payload_to_action(payload, state)
        except Exception:
            return OraclePolicy(self.nominal_capacity_ah).select_action(state)

    def _payload_to_action(self, payload: dict[str, Any], state: BatteryState) -> ProtocolAction:
        mode = str(payload["mode"])
        c_rate = float(payload.get("c_rate", 0.0))
        duration_min = int(payload.get("duration_min", 10))
        target_soc = payload.get("target_soc")
        voltage_limit_v = payload.get("voltage_limit_v")
        if mode not in {"charge", "discharge", "rest"}:
            raise ValueError("invalid mode")

        # A safe mid-SOC rest does not advance the target cycle budget. Treat it
        # as an unusable LLM action and keep validation moving with the oracle.
        if mode == "rest" and state.temperature_c <= 40.0 and 0.18 < state.soc < 0.96:
            return OraclePolicy(self.nominal_capacity_ah).select_action(state)

        c_rate = max(0.0, min(c_rate, 1.5))
        return ProtocolAction(
            action_id="llm_json",
            mode=mode,
            c_rate=c_rate,
            duration_min=max(5, min(duration_min, 90)),
            target_soc=None if target_soc is None else max(0.05, min(float(target_soc), 0.99)),
            voltage_limit_v=None if voltage_limit_v is None else max(2.8, min(float(voltage_limit_v), 4.25)),
            current_limit_a=c_rate * self.nominal_capacity_ah,
        )
