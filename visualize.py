import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_curve_predictions(model, x, target, device=None,
                           plot_noisy_segment=True,
                           noisy_segment_slice=(356+50, 356*2-50),
                           max_plots=4):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    if not isinstance(x, torch.Tensor):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x.to(device)
    if not isinstance(target, torch.Tensor):
        y_t = torch.tensor(target, dtype=torch.float32, device=device)
    else:
        y_t = target.to(device)

    if x_t.dim() == 1:
        x_b = x_t.unsqueeze(0)
    else:
        x_b = x_t
    if y_t.dim() == 1:
        y_b = y_t.unsqueeze(0)
    else:
        y_b = y_t

    N = min(x_b.shape[0], max_plots)

    with torch.no_grad():
        mu_b, sigma_b = model(x_b)           # expect shapes (N, T), (N, T)
        mu_np = mu_b.detach().cpu().numpy()
        sigma_np = sigma_b.detach().cpu().numpy()
        x_np = x_b.detach().cpu().numpy()
        y_np = y_b.detach().cpu().numpy()

    cols = min(4, N)
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
    axes = axes.flatten()

    for i in range(N):
        ax = axes[i]
        mean = mu_np[i].flatten()
        std = sigma_np[i].flatten()
        targ = y_np[i].flatten()

        T = mean.shape[0]
        x_axis = np.arange(T)

        ax.fill_between(x_axis, mean - std, mean + std, color='C0', alpha=0.3, label='±1σ (model)')
        ax.fill_between(x_axis, mean - 2*std, mean + 2*std, color='C0', alpha=0.15, label='±2σ (model)')
        ax.plot(x_axis, mean, color='C0', lw=1.5, label='pred_mean')

        ax.plot(x_axis, targ, color='red', lw=1.5, label='target')

        if plot_noisy_segment:
            s0, s1 = noisy_segment_slice
            if x_np.shape[1] > s0 and s0 < s1:
                seg = x_np[i, s0: s1]
                seg_x = np.arange(s0, s0 + len(seg))
                ax.plot(seg, color='orange', label='noisy input segment')
        ax.set_title(f"sample {i}")
        ax.legend(loc='upper right', fontsize='small')

        ymin = min((mean - 2*std).min(), targ.min())
        ymax = max((mean + 2*std).max(), targ.max())
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def plot_predictions_vs_targets(model, x, y, device='cpu', plot_real_mean=True):
    model = model.to(device)
    model.eval()

    if not isinstance(x, torch.Tensor):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x.to(device)
    if not isinstance(y, torch.Tensor):
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
    else:
        y_t = y.to(device)

    with torch.no_grad():
        y_pred = model(x_t).detach().cpu().numpy().squeeze()  # shape (N,)
    y_np = y_t.detach().cpu().numpy()                         # shape (N, y_dim)
    N = y_np.shape[0]

    cols = min(4, N)
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    axes = axes.flatten()

    for i in range(N):
        ax = axes[i]
        yi = y_np[i]
        x_axis = np.arange(yi.shape[-1])

        real_mean = yi.mean()
        pred_mean = float(y_pred[i])

        r = yi.max() - yi.min()
        if r <= 0:
            r = 1.0
        pad = max(0.3 * r, 0.001)
        y_min = yi.min() - pad
        y_max = yi.max() + pad

        ax.plot(x[i][:630])
        ax.plot(x_axis, yi, color='tab:blue', lw=1.5, label='target curve')
        ax.hlines(pred_mean, x_axis[0], x_axis[-1], colors='tab:red', lw=2.0, label='pred mean')
        if plot_real_mean:
            ax.hlines(real_mean, x_axis[0], x_axis[-1], colors='tab:green', lw=1.5, linestyle='--', label='real mean')

        ax.set_ylim(y_min, y_max)
        ax.set_title(f"sample {i}  pred={pred_mean:.3f}  real_mean={real_mean:.3f}")
        ax.legend(loc='upper right', fontsize='small')

    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig