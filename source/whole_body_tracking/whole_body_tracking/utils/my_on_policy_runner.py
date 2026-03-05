import os
import pathlib

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            normalizer = getattr(self, "obs_normalizer", None)
            export_policy_as_onnx(self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        # Video logging state (same as RMA WandbSummaryWriter)
        self._saved_videos = {}
        self._video_fps = 50

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        print(f"[SAVE] called at iter {self.current_learning_iteration}, logger_type={getattr(self, 'logger_type', 'NOT SET')}")
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=None, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

            self._log_video_files()

    def _log_video_files(self, log_name: str = "Video", video_subdir: str = "videos"):
        """Scan for completed .mp4 files in log_dir/videos and upload to wandb."""
        log_dir = self.log_dir
        print(f"[VIDEO] _log_video_files called, log_dir={log_dir}, iter={self.current_learning_iteration}")
        if log_dir is None:
            print("[VIDEO] log_dir is None, skipping")
            return
        video_dir = pathlib.Path(os.path.join(log_dir, video_subdir))
        if not video_dir.exists():
            print(f"[VIDEO] video_dir {video_dir} does not exist, skipping")
            return
        videos = list(video_dir.rglob("*.mp4"))
        print(f"[VIDEO] found {len(videos)} videos: {[str(v) for v in videos]}")
        for video in videos:
            video_name = str(video)
            video_size_kb = os.stat(video_name).st_size / 1024
            if video_name not in self._saved_videos:
                self._saved_videos[video_name] = {"size": video_size_kb, "recorded": False, "steps": 0}
                print(f"[VIDEO] new video {video_name}, size={video_size_kb:.1f}KB")
            else:
                video_info = self._saved_videos[video_name]
                if video_info["recorded"]:
                    continue
                elif video_info["size"] == video_size_kb and video_size_kb > 100:
                    if video_info["steps"] >= 1:
                        print(f"[VIDEO] uploading {video_name} to wandb at iter {self.current_learning_iteration}")
                        wandb.log({log_name: wandb.Video(video_name, fps=self._video_fps)})
                        self._saved_videos[video_name]["recorded"] = True
                        print(f"[VIDEO] upload done for {video_name}")
                    else:
                        video_info["steps"] += 1
                        print(f"[VIDEO] {video_name} stable, steps={video_info['steps']}")
                else:
                    print(f"[VIDEO] {video_name} size changed {video_info['size']:.1f}KB -> {video_size_kb:.1f}KB or too small, resetting")
                    self._saved_videos[video_name]["size"] = video_size_kb
                    self._saved_videos[video_name]["steps"] = 0