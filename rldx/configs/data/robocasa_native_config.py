"""RoboCasa-native modality configuration matching GrootRoboCasaEnv output."""
import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

robocasa_native = {
    "state": ModalityConfig(
        modality_keys=[
            "base_position",
            "base_rotation",
            "end_effector_position_absolute",
            "end_effector_position_relative",
            "end_effector_rotation_absolute",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "gripper_qvel",
            "joint_position",
            "joint_position_cos",
            "joint_position_sin",
            "joint_velocity",
        ],
        delta_indices=[0],
    ),
    "action": ModalityConfig(
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        ],
        delta_indices=list(range(16)),
        action_configs=[
            ActionConfig(type=ActionType.EEF, rep=ActionRepresentation.DELTA, format=ActionFormat.DEFAULT),
            ActionConfig(type=ActionType.EEF, rep=ActionRepresentation.DELTA, format=ActionFormat.DEFAULT),
            ActionConfig(type=ActionType.NON_EEF, rep=ActionRepresentation.ABSOLUTE, format=ActionFormat.DEFAULT),
            ActionConfig(type=ActionType.NON_EEF, rep=ActionRepresentation.DELTA, format=ActionFormat.DEFAULT),
            ActionConfig(type=ActionType.NON_EEF, rep=ActionRepresentation.ABSOLUTE, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        modality_keys=["annotation.human.task_description"],
        delta_indices=[0],
    ),
    "video": ModalityConfig(
        modality_keys=["left_view", "right_view", "wrist_view"],
        delta_indices=[-6, -4, -2, 0],
    ),
}

register_modality_config(robocasa_native, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(robocasa_native), EmbodimentTag.NEW_EMBODIMENT)
