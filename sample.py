import argparse
from pathlib import Path
import torch

from model import DiT, DiTConfig
from train import pick_device
from data import visualize_frames

@torch.no_grad()
def sample(model, labels, n_steps=50, cfg=2.0, seed=0):
    was_training = model.training
    model.eval()

    B = labels.shape[0]
    device = labels.device
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    x = torch.randn(B, 1, model.T, model.H, model.W, generator=g).to(device)

    cfg_in = torch.full_like(labels, model.label_embedder.null_class)
    y_in = torch.cat([labels, cfg_in]).to(device)
    
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B, ), i * dt).to(device)
        v = model(torch.cat([x, x]), y_in, torch.cat([t, t]))
        vc, vu = v.chunk(2)
        x = x + dt * (vu + cfg * (vc - vu))

    model.train(was_training)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = pick_device("auto")

    ckpt = torch.load(args.ckpt, map_location=device)

    model = DiT(DiTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])      # live weights, not ema
    T, H, W = model.T, model.H, model.W

    labels = torch.arange(10)
    labels = labels.to(device)

    frames = sample(model, labels)

    gif_dir = Path.cwd() / "frames"
    gif_dir.mkdir(exist_ok=True, parents=True)

    frames = frames.squeeze(1)                     # (10, T, H, W)
    grid = (frames.reshape(2, 5, T, H, W)
                .permute(2, 0, 3, 1, 4)
                .reshape(T, H * 2, W * 5))
    visualize_frames(grid, gif_dir / "grid.gif")

if __name__ == "__main__":
    main()
