from sklearn.datasets import fetch_openml

import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path

def load_data():
    # 70k x 784
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 1, 28, 28).astype("float32")
    y = y.astype("int64")

    # to tensor
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # interpolate
    X_tensor_inter = F.interpolate(X_tensor, size=(16, 16), mode='bilinear', antialias=True)

    return X_tensor_inter.squeeze(1), y_tensor


def fold(p, bound):
    L = 2 * bound
    q = p % L
    return torch.where(q > bound, L - q, q)


def make_frames(digits, num_frames, v_max = 3, scale=2, chunk_size=5000):
    """Places each digit on a canvas `scale` times larger and moves it at a constant
    velocity for `num_frames` frames, bouncing off the edges.
    Returns (N, num_frames, H, W)."""
    N, w, h = digits.size() # w indexes rows, h columns (digits are square)
    W, H = w * scale, h * scale # frame size
    Br, Bc = w * (scale - 1), h * (scale - 1) # travel bounds

    start_w = torch.randint(0, Br + 1, (N, )) # (N, )
    start_h = torch.randint(0, Bc + 1, (N, )) # (N, )
    start = torch.stack((start_w, start_h), dim=1) # (N, 2)

    # sample v from grid of [-v_max,v_max]**2
    M = 2 * v_max + 1 # -v_max <= v <= v_max
    origin = (M * M - 1) // 2 # (0, 0) origin
    i = torch.randint(0, M * M - 1, (N,)) # samples
    i = i + (i >= origin).long() # make sure no origin element exists
    v = torch.stack((i // M - v_max, i % M - v_max), dim=1) # roll to grid

    t = torch.arange(num_frames) # (num_frames,)
    pr = (start[:, 0:1] + v[:, 0:1] * t) # (N, num_frames) - start and v are (N,2)
    pc = (start[:, 1:2] + v[:, 1:2] * t) # (N, num_frames) - start and v are (N,2)

    pr = fold(pr, Br) # (N, num_frames)
    pc = fold(pc, Bc) # (N, num_frames)

    ra = torch.arange(w)
    ca = torch.arange(h)

    t_idx = torch.arange(num_frames)[None, :, None, None] # (1, num_frames, 1, 1)
    frames = torch.zeros(N, num_frames, W, H, dtype=torch.float32)

    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        n_idx = torch.arange(i, j)[:, None, None, None]
        rows  = pr[i:j, :, None, None] + ra[None, None, :, None]
        cols  = pc[i:j, :, None, None] + ca[None, None, None, :]
        frames[n_idx, t_idx, rows, cols] = digits[i:j, None]
    
    return frames


def load_frames(num_frames, v_max, scale):
    X, y = load_data()
    frames = make_frames(X, num_frames, v_max, scale)

    frames = frames / 255.0 * 2 - 1        # [-1, 1]
    frames = frames.unsqueeze(1)           # (N, 1, num_frames, H, W)

    return frames, y


def visualize_frames(frames, location, zoom=4):
    frames_u8 = ((frames.detach().cpu().clamp(-1, 1) + 1) / 2 * 255)
    imgs = [Image.fromarray(f) for f in frames_u8.type(torch.uint8).numpy()]
    imgs = [img.resize((img.width * zoom, img.height * zoom), Image.NEAREST) for img in imgs]
    imgs[0].save(location, save_all=True, append_images=imgs[1:], duration=160, loop=0)


def main():
    num_frames, v_max, scale = 8, 3, 2
    frames, y = load_frames(num_frames, v_max, scale)

    gif_dir = Path.cwd() / "illustrations"
    Path.mkdir(gif_dir, exist_ok=True, parents=True)

    w, h = frames.size()[-2:]

    # find first digits for examples
    match = (y[None, :] == y.unique()[:, None])   # (10, N) bool
    idx = match.int().argmax(dim=1)            # (10,) first True in each row

    frames_example = frames[idx].reshape(2, 5, num_frames, w, h).permute(2, 0, 3, 1, 4).reshape(num_frames, w*2, h*5) # (num_frames, w*2, h*5)
    visualize_frames(frames_example, gif_dir / "grid.gif")

if __name__ == "__main__":
    main()
