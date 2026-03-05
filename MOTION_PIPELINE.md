# Motion Retargeting → Training Pipeline

## Step 1: Retarget Motion
From `spot-imitation/retarget_motion/`:
```bash
cd retarget_motion
python3 retarget_motion.py --config retarget_config_spot --input_dir data --output_dir output_spot
```
Outputs `.txt` files to `retarget_motion/output_spot/`.

## Step 2: Convert TXT → CSV
From `spot-imitation/`:
```bash
python scripts/txt_to_csv.py retarget_motion/output_spot motion_imitation/data
```
Outputs `.csv` files to `motion_imitation/data/`.

## Step 3: SCP CSV from Server to Local
From your local machine:
```bash
scp toberoi@robovisionc:/home/toberoi/spot-imitation/motion_imitation/data/<motion>.csv /Users/tejasoberoi/whole_body_tracking_amrl/
```
e.g. `scp toberoi@robovisionc:/home/toberoi/spot-imitation/motion_imitation/data/pace.csv /Users/tejasoberoi/whole_body_tracking_amrl/`

## Step 4: Rsync CSV to Server
From your local machine (only the `<motion>.csv`, nothing else):
```bash
rsync /Users/tejasoberoi/whole_body_tracking_amrl/<motion>.csv toberoi@robovisionc:/home/toberoi/whole_body_tracking_amrl/
```
e.g. `rsync /Users/tejasoberoi/whole_body_tracking_amrl/pace.csv toberoi@robovisionc:/home/toberoi/whole_body_tracking_amrl/`

## Step 4.5: Enter IsaacSim Container
SSH into `robovisionc`, then make sure to configure the IsaacSim container. Enter it with:
```bash
./container enter <container_name>
```
Steps 5 onward must be run from inside this IsaacLab container, in the `/home/toberoi/whole_body_tracking_amrl/` repo.

## Step 5: Convert CSV → NPZ
SSH into `robovisionc`, then from `/home/toberoi/whole_body_tracking_amrl/`:
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/csv_to_npz_spot.py \
    --input_file <motion>.csv \
    --input_fps 30 \
    --output_name <motion_name> \
    --headless
```
e.g. `--input_file pace.csv --output_name pace`

Output NPZ is saved to:
```
source/whole_body_tracking/whole_body_tracking/assets/spot/motions/<motion_name>.npz
```

## Step 6: Point Training Config to New NPZ (optional — skip if using registry)
If not using the wandb registry, edit `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/spot/flat_env_cfg.py`, line 19:
```python
self.commands.motion.motion_file = f"{ASSET_DIR}/spot/motions/<motion_name>.npz"
```
e.g. `f"{ASSET_DIR}/spot/motions/pace.npz"`

## Step 7: Run Training
From `/home/toberoi/whole_body_tracking_amrl/`, using the wandb registry (recommended — skips step 6):
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task Tracking-Flat-Spot-v0 \
    --num_envs 64 \
    --headless \
    --logger wandb \
    --video \
    --registry_name "tejasoberoi-the-university-of-texas-at-austin-org/wandb-registry-motions/<motion>:latest"
```
e.g. `--registry_name "tejasoberoi-the-university-of-texas-at-austin-org/wandb-registry-motions/pace:latest"`
