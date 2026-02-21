from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.spot import SPOT_ACTION_SCALE, SPOT_WBT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class SpotFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SPOT_WBT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = SPOT_ACTION_SCALE
        self.commands.motion.motion_file = f"{ASSET_DIR}/spot/motions/dog_pace.npz"
        self.commands.motion.anchor_body_name = "body"
        self.commands.motion.body_names = [
            "body",
            "fl_hip",
            "fr_hip",
            "hl_hip",
            "hr_hip",
            "fl_uleg",
            "fr_uleg",
            "hl_uleg",
            "hr_uleg",
            "fl_lleg",
            "fr_lleg",
            "hl_lleg",
            "hr_lleg",
            "fl_foot",
            "fr_foot",
            "hl_foot",
            "hr_foot",
        ]

        # Override base_com: base class hardcodes "torso_link" (G1-specific)
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="body")

        # Override ee_body_pos termination: base class hardcodes G1 ankle/wrist links
        self.terminations.ee_body_pos.params["body_names"] = [
            "fl_foot",
            "fr_foot",
            "hl_foot",
            "hr_foot",
        ]

        # Override undesired_contacts: base class regex excludes G1 ankle/wrist links
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!fl_foot$)(?!fr_foot$)(?!hl_foot$)(?!hr_foot$).+$"
            ],
        )
