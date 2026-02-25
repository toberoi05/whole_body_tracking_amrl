from isaaclab_assets.robots.spot import SPOT_CFG

SPOT_WBT_CFG = SPOT_CFG.replace(soft_joint_pos_limit_factor=0.9)
"""Spot config for whole-body tracking: SPOT_CFG with soft joint limit factor."""

# Action scale: 0.25 * effort_limit / stiffness = 0.25 * 45.0 / 60.0
SPOT_ACTION_SCALE = 0.1875
