"""Modality config for RoboCasa MimicGen datasets with flat 158-dim state."""

import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

robocasa_flat = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_view", "right_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["full_state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["full_action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(robocasa_flat, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(robocasa_flat), EmbodimentTag.NEW_EMBODIMENT)
