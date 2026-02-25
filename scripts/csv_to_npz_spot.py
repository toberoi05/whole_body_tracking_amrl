"""Convert a Spot motion CSV to NPZ format for whole_body_tracking.

CSV format (19 columns): x, y, z, qx, qy, qz, qw, then 12 joint angles in
PyBullet enumeration order from spot_noarm_feet.urdf:
  fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn

Usage:
    python scripts/csv_to_npz_spot.py --input_file dog_pace.csv --input_fps 30 --output_name dog_pace --headless
"""

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert Spot motion CSV to NPZ.")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"))
parser.add_argument("--output_name", type=str, required=True)
parser.add_argument("--output_fps", type=int, default=50)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from isaaclab_assets.robots.spot import SPOT_CFG

SPOT_WBT_CFG = SPOT_CFG.replace(soft_joint_pos_limit_factor=0.9)

# Output path: source/whole_body_tracking/whole_body_tracking/assets/spot/motions/<output_name>.npz
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NPZ_OUTPUT_DIR = (
    _REPO_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "assets" / "spot" / "motions"
)

# Joint order from PyBullet enumeration of spot_noarm_feet.urdf
SPOT_JOINT_NAMES = [
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = SPOT_WBT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f}s, frames: {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1))
        print(f"Motion interpolated: {self.input_frames} frames @ {self.input_fps}fps -> {self.output_frames} frames @ {self.output_fps}fps")

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx:self.current_idx + 1],
            self.motion_base_rots[self.current_idx:self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx:self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx:self.current_idx + 1],
            self.motion_dof_poss[self.current_idx:self.current_idx + 1],
            self.motion_dof_vels[self.current_idx:self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return state, reset_flag


def run_simulator(sim, scene):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(SPOT_JOINT_NAMES, preserve_order=True)[0]

    log = {"fps": [args_cli.output_fps], "joint_pos": [], "joint_vel": [],
           "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}
    file_saved = False

    while simulation_app.is_running():
        (motion_base_pos, motion_base_rot, motion_base_lin_vel, motion_base_ang_vel,
         motion_dof_pos, motion_dof_vel), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)

            # Save permanently (used as local fallback)
            output_path = _NPZ_OUTPUT_DIR / f"{args_cli.output_name}.npz"
            np.savez(str(output_path), **log)
            print(f"[INFO]: Motion saved locally to {output_path}")

            # Also save to /tmp for wandb upload
            np.savez("/tmp/motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=COLLECTION)
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motions"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
