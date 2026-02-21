import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

SPOT_WBT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "[fh]l_hx": 0.1,   # all left hip_x
            "[fh]r_hx": -0.1,  # all right hip_x
            "f[rl]_hy": 0.9,   # front hip_y
            "h[rl]_hy": 1.1,   # hind hip_y
            ".*_kn": -1.5,     # all knees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "spot_hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit_sim=45.0,
            velocity_limit_sim=10.0,
            stiffness=60.0,
            damping=1.5,
        ),
        "spot_knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit_sim=45.0,
            velocity_limit_sim=10.0,
            stiffness=60.0,
            damping=1.5,
        ),
    },
)

SPOT_ACTION_SCALE = {}
for a in SPOT_WBT_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            SPOT_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
