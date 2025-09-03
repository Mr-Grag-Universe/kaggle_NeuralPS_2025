import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


class RegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden=512, dropout=0.1):
        super().__init__()
        # small, well-conditioned blocks with skip connections + batchnorm
        self.fc_in = nn.Linear(input_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden)

        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        # widen then squeeze to allow learning scale offsets from mean
        self.wide = nn.Sequential(
            nn.Linear(hidden, hidden*2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden*2),
            nn.Dropout(dropout),
        )
        self.reduce = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        # final head predicts a single scalar; init bias to dataset mean if known (optional)
        self.head = nn.Linear(hidden, 1)

        # weight init (kaiming for ReLU)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = F.relu(x)

        r = self.block1(x) + x            # short residual
        r = self.block2(r) + r            # another residual
        r_w = self.wide(r)
        r = self.reduce(r_w) + r          # long residual
        out = self.head(r)
        return out.squeeze(-1)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    """Basic residual block for 1D convs (no bottleneck)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # nn.Sequential projection if channels/stride differ

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class ResNet1DRegressor(nn.Module):
    """A small ResNet-like 1D model for regression."""
    def __init__(self, x_dim, layers=(2,2,2), channels=(32,64,128), kernel_size=7, dropout=0.2):
        super().__init__()
        assert len(layers) == len(channels)
        self.in_ch = channels[0]
        # initial conv
        self.conv1 = nn.Conv1d(1, self.in_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_ch)
        self.relu = nn.ReLU(inplace=True)

        # build residual stages
        self.layers = nn.ModuleList()
        ch_pairs = list(zip(channels, layers))
        for i, (ch, n_blocks) in enumerate(ch_pairs):
            stride = 1 if i == 0 else 2  # downsample length every stage except first
            self.layers.append(self._make_stage(self.in_ch, ch, n_blocks, kernel_size, stride))
            self.in_ch = ch

        # global pool + regressor
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _make_stage(self, in_channels, out_channels, blocks, kernel_size, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, x_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (B, 1, x_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for stage in self.layers:
            x = stage(x)
        x = self.global_pool(x)  # (B, C_last, 1)
        out = self.fc(x)         # (B, 1)
        return out.squeeze(-1)   # (B,)

# Example usage:
# model = ResNet1DRegressor(x_dim=2048, layers=(2,2,2), channels=(32,64,128))
# y = model(torch.randn(4, 2048))  # -> (4,)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for 1D conv outputs: input shape (B, C, L)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction), bias=True)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels, bias=True)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        # squeeze: global average pooling over L
        s = x.mean(dim=2)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, C)
        s = s.unsqueeze(2)  # (B, C, 1)
        return x * s  # broadcast (B, C, L)

class ConvSERegressor(nn.Module):
    """1D CNN with SE block, global pooling and regressor"""
    def __init__(self, x_dim, channels=(32, 64, 128), kernel_sizes=(7,5,3), dropout=0.2):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        self.input_proj = nn.Conv1d(1, channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        conv_blocks = []
        in_ch = channels[0]
        for out_ch, k in zip(channels[1:], kernel_sizes[1:]):
            conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.se = SEBlock(channels[-1], reduction=16)
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (B, C, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),               # (B, C)
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, x_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (B, 1, x_dim)
        x = self.input_proj(x)  # (B, C0, L)
        x = self.conv_blocks(x) # (B, C_last, L)
        x = self.se(x)          # SE applied
        x = self.pool(x)        # (B, C_last, 1)
        out = self.fc(x)        # (B, 1)
        return out.squeeze(-1)  # (B,)
    
# Example usage:
# model = ConvSERegressor(x_dim=1024)
# y = model(torch.randn(8, 1024))  # -> (8,)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# -------------------------
# Blocks: SE + Channel+Spatial attention (CBAM-like) + Residual conv block
# -------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x):
        # x: (B, C, L)
        s = x.mean(dim=2)              # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(2)  # (B, C, 1)
        return x * s

class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    def forward(self, x):
        # x: (B, C, L)
        avg = x.mean(dim=2)
        max_ = x.max(dim=2)[0]
        out = self.mlp(avg) + self.mlp(max_)
        scale = torch.sigmoid(out).unsqueeze(2)
        return x * scale

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, L)
        avg = x.mean(dim=1, keepdim=True)   # (B,1,L)
        max_ = x.max(dim=1, keepdim=True)[0]# (B,1,L)
        cat = torch.cat([avg, max_], dim=1) # (B,2,L)
        attn = torch.sigmoid(self.conv(cat))# (B,1,L)
        return x * attn

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention1D(channels, reduction=reduction)
        self.spatial = SpatialAttention1D(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.0, use_bn=True, use_cbam=False):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=not use_bn)
        self.bn1 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=padding, bias=not use_bn)
        self.bn2 = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM1D(out_ch, reduction=8, spatial_kernel=7)

        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.act(out)
        out = self.dropout(out)
        if self.use_cbam:
            out = self.cbam(out)
        return out

# -------------------------
# Pro model: deeper Conv + Residual blocks + SE + pooling + regressor
# -------------------------
class ConvSERegressorV2(nn.Module):
    def __init__(
        self,
        x_dim,
        channels=(32, 64, 128, 256),
        kernels=(7,5,3,3),
        blocks_per_stage=(2,2,2,2),
        dropout=0.2,
        use_cbam=True,
        se_reduction=16,
        final_mlp=(128, 64),
        act=nn.ReLU
    ):
        super().__init__()
        assert len(channels) == len(kernels) == len(blocks_per_stage)
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=kernels[0], padding=kernels[0]//2, bias=False)
        self.input_bn = nn.BatchNorm1d(channels[0])
        self.act = act(inplace=True)
        stages = []
        in_ch = channels[0]
        for idx, (out_ch, k, n_blocks) in enumerate(zip(channels, kernels, blocks_per_stage)):
            stride = 1 if idx == 0 else 2  # downsample every stage except first
            # first block in stage may change channels / stride
            stages.append(ResidualConvBlock1D(in_ch, out_ch, kernel=k, stride=stride, dropout=dropout, use_cbam=False))
            for _ in range(1, n_blocks):
                stages.append(ResidualConvBlock1D(out_ch, out_ch, kernel=k, stride=1, dropout=dropout, use_cbam=False))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # SE and optional CBAM at top
        self.se = SEBlock1D(channels[-1], reduction=se_reduction)
        self.cbam_top = CBAM1D(channels[-1], reduction=8, spatial_kernel=7) if use_cbam else nn.Identity()

        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (B, C, 1)

        mlp_layers = []
        in_dim = channels[-1]
        for h in final_mlp:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(act(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*mlp_layers)

        # initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, x_dim) or (B,1,x_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.act(x)
        x = self.stages(x)
        x = self.se(x)
        x = self.cbam_top(x)
        x = self.pool(x)      # (B, C, 1)
        x = x.squeeze(2)      # (B, C)
        out = self.fc(x)      # (B, 1)
        return out.squeeze(-1) # (B,)

# -------------------------
# Small synthetic dataset for demo
# -------------------------
class SyntheticSpectrumDataset(Dataset):
    """Generates synthetic spectra where target = integral (approx) of spectrum plus noise."""
    def __init__(self, N=2000, x_dim=1024, noise_std=0.05, seed=0):
        super().__init__()
        self.N = N
        self.x_dim = x_dim
        torch.manual_seed(seed)
        self.data = []
        self.targets = []
        freqs = torch.linspace(0, 1, x_dim)
        for i in range(N):
            # random mixture of Gaussians peaks
            n_peaks = torch.randint(1, 6, (1,)).item()
            spectrum = torch.zeros(x_dim)
            for _ in range(n_peaks):
                amp = torch.rand(1).item() * 2.0
                center = torch.rand(1).item()
                width = 0.01 + torch.rand(1).item() * 0.1
                spectrum += amp * torch.exp(-0.5 * ((freqs - center) / width)**2)
            # smooth baseline and noise
            baseline = 0.1 * torch.sin(2 * math.pi * freqs * (1 + torch.rand(1).item()*5))
            spectrum = spectrum + baseline
            target = spectrum.sum().item() / x_dim  # average value as target
            spectrum += noise_std * torch.randn_like(spectrum)
            self.data.append(spectrum)
            self.targets.append(target)
        self.data = torch.stack(self.data).float()
        self.targets = torch.tensor(self.targets).float()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# -------------------------
# Training loop demo
# -------------------------
def train_demo(
    x_dim=1024,
    batch_size=32,
    epochs=10,
    lr=1e-3,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SyntheticSpectrumDataset(N=2000, x_dim=x_dim)
    n_val = int(0.1 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvSERegressorV2(
        x_dim=x_dim,
        channels=(32, 64, 128, 256),
        kernels=(7,5,3,3),
        blocks_per_stage=(2,2,2,2),
        dropout=0.2,
        use_cbam=True,
        se_reduction=16,
        final_mlp=(128, 64)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch:02d}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}  LR: {scheduler.get_last_lr()[0]:.2e}")

    return model

# Example run:
# model = train_demo(x_dim=1024, epochs=12, batch_size=64)



def train_mean_model(
    model,
    X, y,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    val_split=0.1,
    device=None,
    loss_fn=None,
    scheduler=None,
):
    """
    X: np.ndarray or torch.Tensor (N, d)
    y: np.ndarray or torch.Tensor (N, k) — для регрессии или one-hot/labels для классификации
    loss_fn: если None, будет MSELoss (регрессия). Для классификации используйте CrossEntropyLoss и подавайте y как class indices.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Преобразуем в тензоры
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    N = X.shape[0]
    # Разделение на train/val
    if val_split and val_split > 0:
        val_size = int(N * val_split)
        train_size = N - val_size
        train_ds = TensorDataset(X[:train_size], y[:train_size])
        val_ds = TensorDataset(X[train_size:], y[train_size:])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_ds = TensorDataset(X, y)
        val_loader = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Loss
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = {'val' : [], 'train' : []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            # Если классификация с CrossEntropyLoss: preds.shape = (B, C), yb dtype long with class indices.
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
        if scheduler is None:
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader.dataset)
        losses['train'].append(float(avg_train_loss))

        # Валидация
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    val_loss += loss_fn(preds, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            # current_lr = None if scheduler is None else scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:03d} | train_loss: {avg_train_loss:.7f} | val_loss: {avg_val_loss:.7f}")
            losses['val'].append(float(avg_val_loss))
        else:
            print(f"Epoch {epoch:03d} | train_loss: {avg_train_loss:.7f}")

    return model, losses