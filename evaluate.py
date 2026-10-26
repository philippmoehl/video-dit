"""Label accuracy and motion of sampled clips.

    python evaluate.py --ckpt checkpoints/step_040000.pt

Trains a small frame classifier on the cached training clips, samples clips for
every digit from the checkpoint, and reports how often the classifier sees the
digit that was asked for, and how often the digit moves by the rule the data was
made with: constant velocity, bouncing off the edges.
"""

import argparse

import torch
from torch import nn

from data import fold
from model import DiT, DiTConfig
from sample import sample
from train import pick_device


def train_classifier(frames, labels, device, iters=1500, batch_size=128):
    """frames: (N, 1, T, H, W). Learns to classify single frames into 10 digits."""
    clf = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1),
        nn.Flatten(), nn.Linear(64, 10),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    N, T = frames.size(0), frames.size(2)
    for _ in range(iters):
        idx = torch.randint(0, N, (batch_size,))
        t = torch.randint(0, T, (batch_size,))
        x = frames[idx, :, t].to(device)            # (batch, 1, H, W)
        loss = nn.functional.cross_entropy(clf(x), labels[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return clf.eval()


@torch.no_grad()
def clip_scores(clf, clips, labels):
    """clips: (B, 1, T, H, W). Returns (label accuracy, frame agreement, clip predictions)."""
    B, _, T, H, W = clips.size()
    probs = clf(clips.transpose(1, 2).reshape(B * T, 1, H, W)).softmax(-1).view(B, T, 10)
    pred = probs.mean(1).argmax(-1)                             # (B,) one digit per clip
    accuracy = (pred == labels).float().mean().item()
    agreement = (probs.argmax(-1) == pred[:, None]).float().mean().item()   # same digit in every frame
    return accuracy, agreement, pred


def track(clips, k):
    """clips: (B, 1, T, H, W). For each pair of consecutive frames, the integer shift in
    [-k, k] per axis that overlays the first frame best on the second. Returns (B, T-1, 2)."""
    x = (clips.squeeze(1) + 1) / 2                                  # (B, T, H, W) in [0, 1]
    H, W = x.shape[-2:]
    prev = nn.functional.pad(x[:, :-1], (k, k, k, k))               # (B, T-1, H+2k, W+2k)
    shifts = [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1)]
    overlap = torch.stack([(prev[..., k - dr:k - dr + H, k - dc:k - dc + W] * x[:, 1:]).sum((-2, -1))
                           for dr, dc in shifts], -1)               # (B, T-1, (2k+1)**2)
    return torch.tensor(shifts, device=x.device)[overlap.argmax(-1)]


@torch.no_grad()
def motion_scores(clips, v_max, bound):
    """clips: (B, 1, T, H, W). The data moves each digit at a constant velocity in [-v_max, v_max]
    per axis and folds it back at 0 and `bound`. Fits the closest such path to each clip, per axis.
    Returns (share of frame steps on that path, share of clips with every step on it)."""
    T = clips.size(2)
    steps = track(clips, v_max + 1)                                 # (B, T-1, 2)

    # every path one axis can take: start in [0, bound], velocity in [-v_max, v_max]
    start, v = torch.meshgrid(torch.arange(bound + 1), torch.arange(-v_max, v_max + 1), indexing="ij")
    paths = fold(start.reshape(-1, 1) + v.reshape(-1, 1) * torch.arange(T), bound)   # (K, T)
    path_steps = paths.diff(dim=1).to(clips.device)                 # (K, T-1)

    on_path = torch.ones(steps.shape[:2], dtype=torch.bool, device=clips.device)
    for axis in range(2):
        hits = steps[:, None, :, axis] == path_steps[None]          # (B, K, T-1)
        best = hits.sum(-1).argmax(-1)                              # (B,) closest path
        on_path &= hits[torch.arange(len(best), device=clips.device), best]
    return on_path.float().mean().item(), on_path.all(-1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true", help="sample from the EMA weights")
    parser.add_argument("--cache", default="cache/frames_T8_v3_s2_n20000.pt",
                    help="frame cache written by train.py")
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()

    device = pick_device("auto")
    torch.manual_seed(0)

    blob = torch.load(args.cache)
    frames, labels = blob["frames"], blob["labels"]
    n_test = 2000
    clf = train_classifier(frames[:-n_test], labels[:-n_test], device)

    test_x, test_y = frames[-n_test:].to(device), labels[-n_test:].to(device)
    real_acc, real_agree, _ = clip_scores(clf, test_x, test_y)
    print(f"real clips     label accuracy {real_acc:.3f}  frame agreement {real_agree:.3f}  ({n_test} held-out clips)")

    v_max, scale = 3, 2                     # data settings of the default --cache, as in train.py
    H = frames.size(-2)
    bound = H - H // scale                  # corner of the H // scale digit moves in [0, bound]
    real_steps, real_clips = motion_scores(test_x, v_max, bound)
    print(f"               motion {real_steps:.3f}  clips fully on path {real_clips:.3f}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = DiT(DiTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["ema" if args.ema else "model"])

    clips, wanted = [], []
    for i in range(args.per_class):
        batch = torch.arange(10, device=device)
        clips.append(sample(model, batch, seed=i).clamp(-1, 1))
        wanted.append(batch)
    clips, wanted = torch.cat(clips), torch.cat(wanted)

    acc, agree, pred = clip_scores(clf, clips, wanted)
    weights = "ema" if args.ema else "live"
    print(f"sampled clips  label accuracy {acc:.3f}  frame agreement {agree:.3f}  "
          f"({len(wanted)} clips, step {ckpt['step']}, {weights} weights)")
    steps_ok, clips_ok = motion_scores(clips, v_max, bound)
    print(f"               motion {steps_ok:.3f}  clips fully on path {clips_ok:.3f}")
    per_digit = [(pred[wanted == d] == d).float().mean().item() for d in range(10)]
    print("per digit      " + "  ".join(f"{d}:{a:.1f}" for d, a in enumerate(per_digit)))


if __name__ == "__main__":
    main()
