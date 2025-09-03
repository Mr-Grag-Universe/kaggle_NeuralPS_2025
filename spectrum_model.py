import torch
import torch.nn as nn
from torch.distributions import Normal
from gpytorch.utils.transforms import inv_softplus
import torch
import numpy as np

class CurveUncertaintyNet(nn.Module):
    def __init__(self,
                 in_features=1000,
                 out_points=283,
                 hidden_dims=(1024, 512, 256),
                 dropout=0.1,
                 max_sigma=0.1, min_sigma=1e-6):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Sequential(
            nn.Linear(prev, prev//2),
            nn.ReLU(inplace=True),
            nn.Linear(prev//2, out_points)
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(prev, prev//2),
            nn.ReLU(inplace=True),
            nn.Linear(prev//2, out_points)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        sigma_raw = self.sigma_head(h)
        sigma = torch.clip(self.softplus(sigma_raw) + self.min_sigma, 0, self.max_sigma)
        return mu, sigma

def gaussian_nll_loss_manual(mu, sigma, target, reduction='mean', eps=1e-8):
    var = sigma.clamp(min=eps) ** 2
    diff2 = (target - mu).pow(2)
    nll_per_elem = 0.5 * (torch.log(2 * torch.pi * var) + diff2 / var)
    nll_per_sample = nll_per_elem.sum(dim=1)
    if reduction == 'mean':
        return nll_per_sample.mean()
    elif reduction == 'sum':
        return nll_per_sample.sum()
    elif reduction == 'none':
        return nll_per_sample
    else:
        raise ValueError

def gaussian_nll_loss_dist(mu, sigma, target, reduction='mean'):
    dist = Normal(mu, sigma.clamp(min=1e-8))
    lp = dist.log_prob(target)
    nll_per_sample = -lp.sum(dim=1)
    if reduction == 'mean':
        return nll_per_sample.mean()
    elif reduction == 'sum':
        return nll_per_sample.sum()
    else:
        return nll_per_sample

def init_sigma_head_to_value(model: CurveUncertaintyNet, desired_sigma: float):
    raw_target = inv_softplus(torch.tensor(max(desired_sigma - model.min_sigma, 1e-12)))
    last_linear = None
    for m in reversed(list(model.sigma_head)):
        if isinstance(m, nn.Linear):
            last_linear = m
            break
    if last_linear is None:
        return
    with torch.no_grad():
        last_linear.weight.fill_(0.0)
        last_linear.bias.fill_(float(raw_target))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def step(self, metric, model):
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        improved = (metric < self.best - self.min_delta) if self.mode == 'min' else (metric > self.best + self.min_delta)
        if improved:
            self.best = metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def evaluate_spectrum_model(model, X, Y, device='cpu', batch_size=256):
    model.eval()
    n = X.shape[0]
    total_loss = 0.0
    total_samples = 0
    sigmas = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size].to(device)
            yb = Y[i:i+batch_size].to(device)
            mu, sigma = model(xb)
            loss_batch = gaussian_nll_loss_manual(mu, sigma, yb, reduction='sum')  # sum over batch
            total_loss += loss_batch.item()
            total_samples += xb.shape[0]
            sigmas.append(float(sigma.mean().cpu().item()))
    avg_loss = total_loss / total_samples
    avg_sigma = float(np.mean(sigmas)) if len(sigmas) else 0.0
    model.train()
    return avg_loss, avg_sigma

def train_spectrum_model(model, train_X, train_Y, val_X, val_Y, device='cpu',
                         lr=1e-4, training_iter=1000, desired_init_noise=5e-3,
                         weight_decay=1e-4, patience=50, min_delta=1e-7,
                         log_every=10, batch_size=None):
    device = device
    model = model.to(device)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)

    init_sigma_head_to_value(model, desired_init_noise)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    earlystop = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')
    losses = []
    val_losses = []
    mean_sigmas = []
    val_sigmas = []

    N = train_X.shape[0]
    batch_size = batch_size or N  # full-batch by default (as in your GP code)

    for it in range(1, training_iter + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        samples = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = train_X[idx]
            yb = train_Y[idx]

            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll_loss_manual(mu, sigma, yb, reduction='mean')
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.shape[0]
            samples += xb.shape[0]

        avg_train_loss = epoch_loss / samples
        with torch.no_grad():
            mu_t, sigma_t = model(train_X[:min(256, N)])
            train_sigma_avg = float(sigma_t.mean().cpu().item())

        val_loss, val_sigma_avg = evaluate_spectrum_model(model, val_X, val_Y, device=device, batch_size=1024)

        losses.append(avg_train_loss)
        val_losses.append(val_loss)
        mean_sigmas.append(train_sigma_avg)
        val_sigmas.append(val_sigma_avg)

        if it % log_every == 0 or it == 1:
            print(f"Iter {it}/{training_iter} - Train NLL: {avg_train_loss:.6f}  Val NLL: {val_loss:.6f}  train_sigma: {train_sigma_avg:.6e}  val_sigma: {val_sigma_avg:.6e}")

        earlystop.step(val_loss, model)
        if earlystop.early_stop:
            print(f"Early stopping at iter {it}. Best val NLL: {earlystop.best:.6f}")
            break

    earlystop.load_best(model)
    model.to(device).eval()
    return model, losses, val_losses, mean_sigmas, val_sigmas