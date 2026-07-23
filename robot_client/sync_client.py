"""Synchronous NZ100 GR00T policy client.

This is the normal request-response inference path provided by GR00T:
send one observation to the GPU policy server and receive one action chunk.
"""

from __future__ import annotations

import numpy as np
from gr00t.policy.server_client import PolicyClient

from robot_client.config import ClientConfig
from robot_client.state_builder import NZ100RobotState


class NZ100SyncClient:
    """Thin wrapper around GR00T's ZMQ PolicyClient for NZ100."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._policy = PolicyClient(
            host=config.server_host,
            port=config.server_port,
            timeout_ms=60000,
        )

    def infer(
        self,
        *,
        top_image: np.ndarray,
        wrist_left_image: np.ndarray,
        robot_state: NZ100RobotState,
        prompt: str | None = None,
        options: dict | None = None,
    ) -> np.ndarray:
        """Return an action chunk with shape ``(action_horizon, 16)``."""

        top_image = _as_uint8_rgb(top_image)
        wrist_left_image = _as_uint8_rgb(wrist_left_image)

        observation = {
            "video": {
                "top": top_image[None, None],
                "wrist_left": wrist_left_image[None, None],
            },
            "state": {
                "left_arm": _batched_state(robot_state.left_joints),
                "left_gripper": _batched_state([robot_state.left_gripper]),
                "right_arm": _batched_state(robot_state.right_joints),
                "right_gripper": _batched_state([robot_state.right_gripper]),
            },
            "language": {
                "annotation.human.task_description": [[
                    self._config.prompt if prompt is None else prompt
                ]],
            },
        }
        actions, _ = self._policy.get_action(observation, options=options)
        action_chunk = _pack_action_chunk(actions)

        if action_chunk.ndim != 2 or action_chunk.shape[-1] != 16:
            raise ValueError(f"Expected action chunk shape (horizon, 16), got {action_chunk.shape}")
        return action_chunk

    def reset(self) -> None:
        self._policy.reset()


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _batched_state(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[None, None]


def _pack_action_chunk(actions: dict[str, np.ndarray]) -> np.ndarray:
    required_keys = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    missing = [key for key in required_keys if key not in actions]
    if missing:
        raise ValueError(f"GR00T action missing keys: {missing}; got {sorted(actions)}")

    left_arm = _single_batch_action(actions["left_arm"], "left_arm")
    left_gripper = _single_batch_action(actions["left_gripper"], "left_gripper")
    right_arm = _single_batch_action(actions["right_arm"], "right_arm")
    right_gripper = _single_batch_action(actions["right_gripper"], "right_gripper")
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)


def _single_batch_action(value: np.ndarray, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"Expected action[{key!r}] shape (1, horizon, dim), got {value.shape}")
    return value[0]
