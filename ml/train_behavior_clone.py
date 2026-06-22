"""
Behavior cloning trainer.

Trains the PPO actor network via supervised imitation of the teacher autopilot,
then saves a standard SB3 .zip that can be loaded directly for PPO warm-start.

Usage:
    python ml/train_behavior_clone.py --demo results/demos/waypoint_demo_001.npz
    python ml/train_behavior_clone.py --demo results/demos/stabilize_demo_001.npz \\
        --epochs 80 --lr 5e-4 --out models/bc/stabilize_bc

Multiple demo files can be combined:
    python ml/train_behavior_clone.py \\
        --demo results/demos/waypoint_demo_001.npz results/demos/waypoint_demo_002.npz

Output: models/bc/<phase>_bc.zip  (SB3 PPO, actor pre-trained)

Warm-start PPO from this checkpoint:
    python train.py --init-from-bc models/bc/waypoint_bc
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.fixed_wing_env import FixedWingEnv


def load_demos(paths: list[str]):
    obs_list, act_list = [], []
    for p in paths:
        d = np.load(p)
        obs_list.append(d["obs"])
        act_list.append(d["actions"])
        n = len(d["obs"])
        print(f"[BC] Loaded {n:,} steps from {p}")
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    return obs, act


def train_bc(
    demo_paths: list[str],
    out_path:   str,
    epochs:     int   = 60,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    val_frac:   float = 0.10,
):
    obs_all, act_all = load_demos(demo_paths)
    n     = len(obs_all)
    split = int(n * (1.0 - val_frac))

    # Shuffle before split
    perm     = np.random.permutation(n)
    obs_all  = obs_all[perm]
    act_all  = act_all[perm]

    obs_train, act_train = obs_all[:split], act_all[:split]
    obs_val,   act_val   = obs_all[split:], act_all[split:]

    print(f"[BC] Total: {n:,}  train: {split:,}  val: {n - split:,}")

    # Build a throw-away env just to get the correct spaces for PPO init
    def _make_env():
        return FixedWingEnv(training_mode=True, curriculum_phase="mixed")

    vec_env = DummyVecEnv([_make_env])
    model   = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs = dict(net_arch=[256, 256]),
        verbose       = 0,
    )

    policy    = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    obs_t_train = torch.FloatTensor(obs_train)
    act_t_train = torch.FloatTensor(act_train)
    obs_t_val   = torch.FloatTensor(obs_val)
    act_t_val   = torch.FloatTensor(act_val)

    best_val   = float("inf")
    n_train    = len(obs_t_train)

    print(f"\n[BC] Training {epochs} epochs  batch={batch_size}  lr={lr:.0e}")
    print(f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}  "
          f"{'ELEV':>6}  {'AIL':>6}  {'RUD':>6}  {'THR':>6}")

    for epoch in range(1, epochs + 1):
        policy.train()
        perm_ep = torch.randperm(n_train)
        ep_losses = []

        for start in range(0, n_train, batch_size):
            idx   = perm_ep[start:start + batch_size]
            obs_b = obs_t_train[idx]
            act_b = act_t_train[idx]

            dist      = policy.get_distribution(obs_b)
            act_pred  = dist.distribution.mean
            loss      = F.mse_loss(act_pred, act_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            ep_losses.append(loss.item())

        scheduler.step()

        # Validation
        policy.eval()
        with torch.no_grad():
            dist_v   = policy.get_distribution(obs_t_val)
            pred_v   = dist_v.distribution.mean
            val_loss = F.mse_loss(pred_v, act_t_val).item()
            mae      = (pred_v - act_t_val).abs().mean(dim=0)

        train_loss = float(np.mean(ep_losses))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:6d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                  f"{mae[0].item():6.3f}  {mae[1].item():6.3f}  "
                  f"{mae[2].item():6.3f}  {mae[3].item():6.3f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            model.save(out_path)

    print(f"\n[BC] Done.  Best val MSE: {best_val:.4f}")
    print(f"[BC] Saved -> {out_path}.zip")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning from teacher demos")
    parser.add_argument("--demo",       nargs="+", required=True,
                        help="Path(s) to .npz demo files")
    parser.add_argument("--out",        default=None,
                        help="Output path (without .zip).  "
                             "Defaults to models/bc/<phase>_bc")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-frac",   type=float, default=0.10)
    args = parser.parse_args()

    # Derive default output name from first demo filename
    if args.out is None:
        base  = os.path.basename(args.demo[0])
        phase = base.split("_demo_")[0] if "_demo_" in base else "bc"
        args.out = os.path.join("models", "bc", f"{phase}_bc")

    train_bc(
        demo_paths = args.demo,
        out_path   = args.out,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        val_frac   = args.val_frac,
    )


if __name__ == "__main__":
    main()
