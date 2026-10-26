"""Flow-matching training for the video DiT.

    python train.py --steps 40000        # real run

Frames are cached to disk after the first run; generating them from MNIST
takes a while. Generation is seeded, so the same settings give the same clips.
"""

import argparse
import copy
from dataclasses import asdict
import time
from pathlib import Path

import torch
from torch import nn

from data import load_frames
from model import DiT, DiTConfig


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_frames(num_frames, v_max, scale, max_clips, cache):
    """Returns (frames, labels) on CPU. frames: (N, 1, T, H, W) in [-1, 1]."""
    cache = Path(cache)
    if cache.exists():
        blob = torch.load(cache)
        frames, labels = blob["frames"], blob["labels"]
        print(f"loaded {frames.size(0)} clips from {cache}")
    else:
        frames, labels = load_frames(num_frames, v_max, scale)
        if max_clips and frames.size(0) > max_clips:
            keep = torch.randperm(frames.size(0))[:max_clips]
            frames, labels = frames[keep], labels[keep]
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"frames": frames, "labels": labels}, cache)
        print(f"generated {frames.size(0)} clips, cached to {cache}")

    gb = frames.numel() * frames.element_size() / 1e9
    print(f"frames {tuple(frames.size())}  {gb:.2f} GB  "
          f"range [{frames.min():.2f}, {frames.max():.2f}]")
    return frames, labels


def flow_matching_loss(model, x1, y):
    """x1: (B, C, T, H, W) in [-1, 1]. Scalar MSE on the velocity."""
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)

    t_b = t.view(B, 1, 1, 1, 1)          # broadcast over C, T, H, W
    xt = (1 - t_b) * x0 + t_b * x1
    target = x1 - x0                     # constant velocity along the path

    pred = model(xt, y, t)               # model takes t as (B,)
    return nn.functional.mse_loss(pred, target)


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-clips", type=int, default=20000,
                    help="0 keeps all 70k (~2.3 GB at float32)")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model_cfg = DiTConfig()
    T, (H, W) = model_cfg.num_frames, model_cfg.img_size
    v_max, scale = 3, 2
    cache = Path(args.cache_dir) / f"frames_T{T}_v{v_max}_s{scale}_n{args.max_clips}.pt"
    frames, labels = get_frames(T, v_max, scale, args.max_clips, cache)
    assert frames.shape[-3:] == (T, H, W), frames.shape

    model = DiT(model_cfg).to(device)

    n_params = sum(q.numel() for q in model.parameters())
    print(f"device={device}  params={n_params/1e6:.2f}M  steps={args.steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    ema = copy.deepcopy(model).eval().requires_grad_(False)

    N = frames.size(0)
    model.train()
    running, t0 = 0.0, time.time()

    for step in range(1, args.steps + 1):
        # frames stay on CPU; one batch at a time goes to the device
        idx = torch.randint(0, N, (args.batch_size,))
        x1 = frames[idx].to(device, non_blocking=True)
        y = labels[idx].to(device, non_blocking=True)

        loss = flow_matching_loss(model, x1, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        ema_update(ema, model, 0.9999)

        loss_val = loss.item()
        running += loss_val

        if step % args.log_every == 0:
            avg = running / args.log_every
            rate = args.log_every / (time.time() - t0)
            print(f"step {step:>6}  loss {avg:.4f}  grad {grad_norm:.2f}  {rate:.1f} it/s")
            running, t0 = 0.0, time.time()

        if step % args.ckpt_every == 0 or step == args.steps:
            payload = {"model_cfg": asdict(model_cfg), "step": step, "model": model.state_dict(),
                       "ema": ema.state_dict(), "args": vars(args)}
            torch.save(payload, Path(args.out) / f"step_{step:06d}.pt")


if __name__ == "__main__":
    main()
