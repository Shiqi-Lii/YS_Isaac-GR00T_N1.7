"""RTC-compatible NZ100 GR00T policy client."""

from __future__ import annotations

import numpy as np

from robot_client.config import ClientConfig
from robot_client.state_builder import NZ100RobotState
from robot_client.sync_client import NZ100SyncClient


class NZ100RTCClient:
    """GR00T client with the same interface used by the RTC runner."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._client = NZ100SyncClient(config)

    def infer(
        self,
        *,
        top_image: np.ndarray,
        wrist_left_image: np.ndarray,
        robot_state: NZ100RobotState,
        previous_chunk: np.ndarray | None = None,
        prefix_len: int | None = None,
        prompt: str | None = None,
    ) -> np.ndarray:
        options = self._make_rtc_options(previous_chunk, prefix_len=prefix_len)
        return self._client.infer(
            top_image=top_image,
            wrist_left_image=wrist_left_image,
            robot_state=robot_state,
            prompt=prompt,
            options=options,
        )

    def reset(self) -> None:
        self._client.reset()

    def _make_rtc_options(
        self, previous_chunk: np.ndarray | None, *, prefix_len: int | None = None
    ) -> dict | None:
        if previous_chunk is None:
            return None

        previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
        if previous_chunk.ndim != 2 or previous_chunk.shape[-1] != 16:
            raise ValueError(f"Expected previous action chunk shape (horizon, 16), got {previous_chunk.shape}")

        prefix_len = int(self._config.rtc_prefix_len if prefix_len is None else prefix_len)
        prefix_len = min(prefix_len, previous_chunk.shape[0])
        if prefix_len <= 0:
            return None

        target_horizon = int(self._config.open_loop_horizon)
        if target_horizon <= 0:
            target_horizon = previous_chunk.shape[0]
        if previous_chunk.shape[0] < target_horizon:
            pad_len = target_horizon - previous_chunk.shape[0]
            previous_chunk = np.concatenate(
                [previous_chunk, np.repeat(previous_chunk[-1:], pad_len, axis=0)],
                axis=0,
            )
        elif previous_chunk.shape[0] > target_horizon:
            previous_chunk = previous_chunk[:target_horizon]

        overlap_steps = int(self._config.rtc_decay_end or target_horizon)
        overlap_steps = max(prefix_len, min(overlap_steps, target_horizon))
        ramp_rate = max(float(self._config.rtc_decay_tau), 1e-6)

        rtc_options = {
            "prev_actions": previous_chunk,
            "action_horizon": target_horizon,
            "rtc_prefix_len": prefix_len,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": prefix_len,
            "rtc_ramp_rate": ramp_rate,
            "rtc_decay_tau": float(self._config.rtc_decay_tau),
            "rtc_decay_end": overlap_steps,
            "rtc_use_vjp": bool(self._config.rtc_use_vjp),
        }

        guidance_weight = float(self._config.rtc_guidance_weight)
        if guidance_weight > 0:
            rtc_options["rtc_guidance_weight"] = guidance_weight

        return {
            "rtc": rtc_options
        }
