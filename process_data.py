import copy
import itertools
import warnings
from typing import List, Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from astropy.stats import sigma_clip
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from scipy.interpolate import CubicSpline
from scipy.signal import resample, resample_poly, savgol_filter
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.linear_model import LinearRegression

from .process_transit import (get_transit_bounds_external,
                              get_transit_bounds_internal)


def calibrate_signal(signal, dead, dark, flat, dt, linear_corr=None):
    def mask_pixels(signal, dead, dark):
        hot = sigma_clip(dark, sigma=5, maxiters=3).mask
        hot_mask = np.tile(hot, (signal.shape[0], 1, 1))
        dead_mask = np.tile(dead, (signal.shape[0], 1, 1))
        signal = np.ma.masked_where(dead_mask | hot_mask, signal)
        return signal
    def apply_linear_corr(clean_signal, linear_corr):
        linear_corr = np.flip(linear_corr, axis=0)
        for x, y in itertools.product(
                    range(clean_signal.shape[1]), 
                    range(clean_signal.shape[2])
                ):
            poli = np.poly1d(linear_corr[:, x, y])
            clean_signal[:, x, y] = poli(clean_signal[:, x, y])
        return clean_signal
    def clean_dark(signal, dead, dark, dt):
        dark = np.ma.masked_where(dead, dark)
        dark = np.tile(dark, (signal.shape[0], 1, 1))
        signal -= dark* dt[:, np.newaxis, np.newaxis]
        return signal
    def correct_flat_field(signal, flat, dead):
        flat = np.ma.masked_where(dead, flat)
        flat = np.tile(flat, (signal.shape[0], 1, 1))
        signal = signal / flat
        return signal

    signal = signal.astype(np.float64)
    signal = mask_pixels(signal, dead, dark)
    if linear_corr is not None:
        signal = apply_linear_corr(signal, linear_corr)
    signal = clean_dark(signal, dead, dark, dt)
    signal = correct_flat_field(signal, flat, dead)
    
    return signal


def get_prepared_cds(signal, binning):
    def get_cds(signal):
        cds = signal[1::2,:,:] - signal[::2,:,:]
        return cds
    def bin_obs(cds_signal,binning):
        n_bins = cds_signal.shape[0] // binning
        cds_binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])
        # cds_binned = cds_signal[:n_bins * binning].reshape(n_bins, binning, -1).mean(axis=1)
        return cds_binned
    cds = get_cds(signal)
    cds = bin_obs(cds, binning=binning)
    return cds

def prepare_signal(signal : np.ndarray,
                   dead=None, dark=None, 
                   linear_corr=None,
                   dt=None,
                   flat=None, 
                   gain=None, offset=None,
                   binning=30,
                   calibrate=True) -> np.ndarray:
    def restore(signal, gain, offset):
        return signal / gain + offset

    signal = signal.astype(np.float64)
    signal = restore(signal, gain, offset)
    signal = calibrate_signal(signal, dead, dark, dt=dt,
                              flat=flat, linear_corr=linear_corr) if calibrate else signal
    cds = get_prepared_cds(signal, binning=binning)

    return cds

def get_stellar_spectrum(cds_signal):
    left, right = get_transit_bounds_external(cds_signal.mean(axis=(1,2)))
    return np.concatenate([cds_signal[:left], cds_signal[right:]]).mean(axis=(0, 1))


def get_flux_drift(flux):
    l, r = get_transit_bounds_external(flux)
    x = np.arange(len(flux)).astype(float)

    left_mask = x <= l
    right_mask = x >= r
    train_mask = left_mask | right_mask
    
    X_train = x[train_mask].reshape(-1, 1)
    y_train = np.asarray(flux)[train_mask]
    
    if X_train.shape[0] < 3:
        X_train = x.reshape(-1, 1)
        y_train = np.asarray(flux)
    
    kernel = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-3, 1e2)) * \
             RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3)) + \
             WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e1))
    
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0,
                                  normalize_y=True,
                                  n_restarts_optimizer=3)
    
    gp.fit(X_train, y_train)
    
    X_pred = x.reshape(-1, 1)
    y_pred, y_std = gp.predict(X_pred, return_std=True)
    
    return y_pred, y_std, gp

def get_flux_drift_ratio(flux):
    drift, _, _ = get_flux_drift(flux)
    return drift / drift.mean()


def smooth_array_with_gp(X_train, y_train, X_pred):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        kernel = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-3, 1e2)) * \
                 RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3)) + \
                 WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0,
                                      normalize_y=True,
                                      n_restarts_optimizer=3)
        
        gp.fit(X_train, y_train)
        y_pred, y_std = gp.predict(X_pred, return_std=True)
    return y_pred


def make_smooth_cap(flux, mineps=1e-8):
    # flux : np.ndarray
    n = len(flux)
    x = np.arange(n).astype(float).reshape(-1, 1)
    ratio = np.ones(n, dtype=float)

    external_l, external_r = get_transit_bounds_external(flux)
    internal_l, internal_r = get_transit_bounds_internal(flux)

    external_l = int(np.clip(external_l, 0, n))
    external_r = int(np.clip(external_r, 0, n))
    internal_l = int(np.clip(internal_l, 0, n))
    internal_r = int(np.clip(internal_r, 0, n))

    # --- 1) flux_drift ---
    mask_out = np.ones(n, dtype=bool)
    if external_l < external_r:
        mask_out[external_l:external_r] = False
    else:
        mask_out[:] = True

    num_out = mask_out.sum()
    if num_out >= 4:
        flux_drift = smooth_array_with_gp(x[mask_out], flux[mask_out], x).ravel()
    elif num_out > 0:
        print(mask_out)
        print(flux[mask_out])
        mean_out = float(np.mean(flux[mask_out]))
        flux_drift = np.full(n, mean_out, dtype=float)
    else:
        k = max(4, n // 8)
        left_idx = slice(0, min(k, n))
        right_idx = slice(max(0, n - k), n)
        combined_idx = np.r_[np.arange(left_idx.start, left_idx.stop), np.arange(right_idx.start, right_idx.stop)]
        if combined_idx.size >= 4:
            # flux_drift_partial = smooth_array_with_gp(x[combined_idx], flux[combined_idx], x[combined_idx]).ravel()
            mean_partial = float(np.mean(flux[combined_idx]))
            flux_drift = np.full(n, mean_partial, dtype=float)
        else:
            flux_drift = np.full(n, float(np.mean(flux)), dtype=float)

    # --- 2) flux_smooth ---
    try:
        flux_smooth = smooth_array_with_gp(x, flux, x).ravel()
    except Exception:
        flux_smooth = flux.astype(float).copy()

    # --- 3) ratio = flux_smooth / flux_drift ---
    denom = flux_drift.copy()
    denom[np.abs(denom) < mineps] = mineps * np.sign(denom[np.abs(denom) < mineps]) if np.any(denom != 0) else mineps
    ratio = (flux_smooth + mineps) / denom
    ratio = np.clip(ratio, 0.0, 1.0)

    # --- 4) ---
    if external_l < external_r:
        ratio[:external_l] = 1.0
        ratio[external_r:] = 1.0
    else:
        # default value
        pass

    # --- 5) internal (bottom) smoothing ---
    if internal_l < internal_r and (internal_r - internal_l) > 3:
        idx_bottom = slice(internal_l, internal_r)
        x_bottom = x[idx_bottom]
        ratio_bottom = ratio[idx_bottom]
        try:
            sm = smooth_array_with_gp(x_bottom, ratio_bottom, x_bottom).ravel()
            ratio[idx_bottom] = sm
        except Exception:
            pass

    # --- 6) linear regression for slides ---
    if external_l < internal_l and (internal_l - external_l) > 3:
        x_left = x[external_l:internal_l]
        y_left = ratio[external_l:internal_l]
        try:
            lr_left = LinearRegression()
            lr_left.fit(x_left, y_left)
            ratio[external_l:internal_l] = lr_left.predict(x_left)
        except Exception:
            pass
    if internal_r < external_r and (external_r - internal_r) > 3:
        x_right = x[internal_r:external_r]
        y_right = ratio[internal_r:external_r]
        try:
            lr_right = LinearRegression()
            lr_right.fit(x_right, y_right)
            ratio[internal_r:external_r] = lr_right.predict(x_right)
        except Exception:
            pass

    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
    ratio = np.clip(ratio, 0.0, 1.0)

    return ratio


def resample_fft_torch(x: torch.Tensor, new_len: int):
    orig_shape = x.shape
    old_len = orig_shape[-1]
    if new_len == old_len:
        return x.clone()
    Xf = torch.fft.rfft(x, n=old_len, dim=-1)
    old_bins = Xf.size(-1)
    new_bins = new_len // 2 + 1
    device = x.device
    dtype = Xf.dtype  # complex dtype
    if new_bins == old_bins:
        Xf_new = Xf
    elif new_bins > old_bins:
        pad_shape = list(Xf.shape)
        pad_shape[-1] = new_bins - old_bins
        pad = torch.zeros(*pad_shape, dtype=dtype, device=device)
        Xf_new = torch.cat([Xf, pad], dim=-1)
    else:
        Xf_new = Xf[..., :new_bins]
    Xf_new = Xf_new * (new_len / old_len)
    y = torch.fft.irfft(Xf_new, n=new_len, dim=-1)
    return y




# ваш torch-ресэмплер (сохранён автоградиент)
def resample_fft_torch(x: torch.Tensor, new_len: int):
    orig_shape = x.shape
    old_len = orig_shape[-1]
    if new_len == old_len:
        return x.clone()
    Xf = torch.fft.rfft(x, n=old_len, dim=-1)
    old_bins = Xf.size(-1)
    new_bins = new_len // 2 + 1
    device = x.device
    dtype = Xf.dtype
    if new_bins == old_bins:
        Xf_new = Xf
    elif new_bins > old_bins:
        pad_shape = list(Xf.shape)
        pad_shape[-1] = new_bins - old_bins
        pad = torch.zeros(*pad_shape, dtype=dtype, device=device)
        Xf_new = torch.cat([Xf, pad], dim=-1)
    else:
        Xf_new = Xf[..., :new_bins]
    Xf_new = Xf_new * (new_len / old_len)
    y = torch.fft.irfft(Xf_new, n=new_len, dim=-1)
    return y

# конвертеры
def _is_torch(x): return isinstance(x, torch.Tensor)
def _to_numpy_cpu(x):
    return x.detach().cpu().numpy() if _is_torch(x) else np.asarray(x)
def _from_numpy_to_ref(np_arr, ref):
    if _is_torch(ref):
        return torch.as_tensor(np_arr, dtype=ref.dtype, device=ref.device)
    return np.asarray(np_arr)

# Вспомогательная обёртка для вызова get_stellar_spectrum и др., возвращающая torch tensor на нужном device
def _stellar_spectrum_torch(cds_signal_t: torch.Tensor) -> torch.Tensor:
    # cds_signal_t: (t,y,l) single
    cds_np = _to_numpy_cpu(cds_signal_t)
    ss_np = get_stellar_spectrum(cds_np)  # numpy (l,)
    return torch.as_tensor(ss_np, dtype=cds_signal_t.dtype, device=cds_signal_t.device)

def _flux_drift_torch(cds_signal_t: torch.Tensor) -> torch.Tensor:
    # returns drift of shape (t,)
    cds_np = _to_numpy_cpu(cds_signal_t)
    flux = cds_signal_t.mean(dim=(1,2)).detach().cpu().numpy()  # use torch mean but convert for function
    drift_np = get_flux_drift_ratio(flux)  # numpy (t,)
    return torch.as_tensor(drift_np, dtype=cds_signal_t.dtype, device=cds_signal_t.device)

def _make_smoothed_cap_torch(flux_t: torch.Tensor) -> torch.Tensor:
    flux_np = flux_t.detach().cpu().numpy()
    sm_np = make_smooth_cap(flux_np)
    return torch.as_tensor(sm_np, dtype=flux_t.dtype, device=flux_t.device)

def _get_transit_bounds_torch(flux_t: torch.Tensor) -> Tuple[int,int]:
    l_np = flux_t.detach().cpu().numpy()
    return get_transit_bounds_internal(l_np)

# main torch implementations for single sample (t,y,l)
def _get_noise_torch_single(cds_signal_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    # cds_signal_t: (t,y,l), target_t: (L_target,) or (l,) depending
    t_len, y_len, l_len = cds_signal_t.shape
    flux = cds_signal_t.mean(dim=(1,2))  # (t,)
    smoothed_cap = _make_smoothed_cap_torch(flux)  # (t,)
    m = smoothed_cap.min()
    M = smoothed_cap.max()
    transit_ratio = 1.0 - (smoothed_cap - m) / (M - m)  # (t,)

    # resample target[1:] to length l_len
    # target_t may be torch or numpy; ensure torch
    target_t = target_t.to(dtype=cds_signal_t.dtype, device=cds_signal_t.device)
    target_trim = target_t[1:]  # (L-1,) maybe
    target_resampled = resample_fft_torch(target_trim.unsqueeze(0), l_len).squeeze(0)  # resample handles last dim
    # transit_spectrum = 1 - outer(transit_ratio, target_resampled)
    transit_spectrum = 1.0 - torch.einsum('i,j->ij', transit_ratio, target_resampled)  # (t, l)

    # stellar_spectrum via numpy helper then tiled
    stellar = _stellar_spectrum_torch(cds_signal_t)  # (l,)
    stellar_spectrum = stellar.unsqueeze(0).expand(t_len, -1)  # (t, l)

    clear_spectrum = transit_spectrum * stellar_spectrum  # (t, l)

    # drift
    drift = _flux_drift_torch(cds_signal_t)  # (t,)
    drift_2d = drift.unsqueeze(1).expand(-1, l_len)  # (t, l)
    noise = cds_signal_t.mean(dim=1) - drift_2d * clear_spectrum  # cds_signal.mean(1) -> (t,l)

    return noise

def _get_target_torch_single(cds_signal_t: torch.Tensor, noise_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    t_len, y_len, l_len = cds_signal_t.shape
    flux = cds_signal_t.mean(dim=(1,2))
    smoothed_cap = _make_smoothed_cap_torch(flux)
    drift = _flux_drift_torch(cds_signal_t)

    m = smoothed_cap.min()
    M = smoothed_cap.max()
    transit_ratio = 1.0 - (smoothed_cap - m) / (M - m)  # (t,)

    spectrum = cds_signal_t.mean(dim=1) - noise_t if noise_t is not None else 0  # (t,l)
    drift_2d = drift.unsqueeze(1).expand(-1, l_len)
    clear_spectrum = spectrum / (torch.abs(drift_2d) + 1e-8)

    stellar = _stellar_spectrum_torch(cds_signal_t)  # (l,)
    stellar_spectrum = stellar.unsqueeze(0).expand(t_len, -1)

    transit_spectrum = 1.0 - torch.clamp(clear_spectrum / stellar_spectrum, 0.0, 1.0)

    transit_spectrum_2d = torch.tile((1.0 - (smoothed_cap - m) / (M - m)).unsqueeze(1), (1, l_len))  # (t,l)
    l_idx, r_idx = _get_transit_bounds_torch(flux)
    target = (transit_spectrum / (transit_spectrum_2d + 1e-12))[l_idx:r_idx, :].mean(dim=0)  # (l,)

    return target

def resample_to_length_fft(x, y_count):
    return resample(x, y_count)


# numpy single implementations — используем ваши оригинальные функции
def _get_noise_numpy_single(cds_signal_np: np.ndarray, target_np: np.ndarray) -> np.ndarray:
    t_len, y_len, l_len = cds_signal_np.shape

    flux = cds_signal_np.mean(axis=(1,2))
    smoothed_cap = make_smooth_cap(flux)
    m, M = smoothed_cap.min(), smoothed_cap.max()
    transit_ratio = (smoothed_cap - m) / (M-m)
    transit_ratio = 1-transit_ratio
    target_resampled = resample_to_length_fft(target_np[1:], l_len)
    transit_spectrum = 1 - np.einsum('i,j->ij', transit_ratio, target_resampled)

    stellar_spectrum = get_stellar_spectrum(cds_signal_np)
    stellar_spectrum = np.tile(stellar_spectrum, (t_len, 1))

    clear_spectrum = transit_spectrum * stellar_spectrum

    drift = get_flux_drift_ratio(flux)
    drift_2d = np.tile(drift, (l_len, 1)).T
    noise = cds_signal_np.mean(1) - drift_2d * clear_spectrum

    return noise

def _get_target_numpy_single(cds_signal_np: np.ndarray, noise_np: np.ndarray) -> np.ndarray:
    t_len, y_len, l_len = cds_signal_np.shape

    flux = cds_signal_np.mean(axis=(1,2))
    smoothed_cap = make_smooth_cap(flux)
    drift = get_flux_drift_ratio(flux)

    m, M = smoothed_cap.min(), smoothed_cap.max()
    transit_ratio = (smoothed_cap - m) / (M-m)
    transit_ratio = 1-transit_ratio

    spectrum = cds_signal_np.mean(1) - noise_np
    drift_2d = np.tile(drift, (l_len, 1)).T
    clear_spectrum = spectrum / (np.abs(drift_2d)+1e-8)

    stellar_spectrum = get_stellar_spectrum(cds_signal_np)
    stellar_spectrum = np.tile(stellar_spectrum, (t_len, 1))

    transit_spectrum = 1 - np.clip(clear_spectrum / stellar_spectrum, 0, 1)

    transit_spectrum_2d = np.tile(transit_ratio, (l_len, 1)).T
    l, r = get_transit_bounds_internal(flux)
    target = (transit_spectrum / transit_spectrum_2d)[l:r,:].mean(0)

    return target

# основной интерфейс, поддерживает numpy / torch и батч
def get_noise(cds_signal, target):
    # numpy path
    if not _is_torch(cds_signal):
        return _get_noise_numpy_single(np.asarray(cds_signal), np.asarray(target))

    # torch path
    cds = cds_signal
    # поддерживаем батч (B,t,y,l) и одиночный (t,y,l)
    if cds.dim() == 4:
        B = cds.shape[0]
        outs = []
        # target может быть (B, L) или (L,)
        for i in range(B):
            tgt_i = target[i] if (_is_torch(target) and target.dim() == 2) else target
            out_i = _get_noise_torch_single(cds[i], tgt_i)
            outs.append(out_i)
        return torch.stack(outs, dim=0)  # (B,t,l)
    else:
        return _get_noise_torch_single(cds, target)

def get_target(cds_signal, noise):
    if not _is_torch(cds_signal):
        return _get_target_numpy_single(np.asarray(cds_signal), np.asarray(noise))

    cds = cds_signal
    if cds.dim() == 4:
        B = cds.shape[0]
        outs = []
        for i in range(B):
            noise_i = noise[i] if (_is_torch(noise) and noise.dim() == 3) else noise
            out_i = _get_target_torch_single(cds[i], noise_i)
            outs.append(out_i)
        return torch.stack(outs, dim=0)  # (B,l)
    else:
        return _get_target_torch_single(cds, noise)


class FluxDriftGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(FluxDriftGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_flux_drift_ratio_gpytorch(flux_t: torch.Tensor) -> torch.Tensor:
    """
    Батчевая версия get_flux_drift_ratio с использованием GPyTorch
    flux_t: torch.Tensor shape (B, T) где B - размер батча, T - временные точки
    возвращает: torch.Tensor shape (B, T)
    """
    assert flux_t.dim() >= 2, "flux_t must be (B, T) or more"
    B, T = flux_t.shape
    device = flux_t.device
    dtype = flux_t.dtype
    
    # Создаем временную ось
    x = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
    
    # Определяем маску для тренировочных данных (вне транзита)
    bounds = [_get_transit_bounds_torch(flux_t[i]) for i in range(B)]
    train_mask = torch.ones((B, T), device=device, dtype=torch.bool)
    
    for i in range(B):
        l, r = bounds[i]
        li = int(max(0, np.floor(l)))
        ri = int(min(T, np.ceil(r)))
        if ri > li:
            train_mask[i, li:ri] = False
    
    # Предварительная подготовка выхода
    drift = torch.zeros_like(flux_t, dtype=dtype, device=device)
    
    for i in range(B):
        # Тренировочные данные для этого элемента батча
        train_x_i = x[i, train_mask[i]].reshape(-1, 1)
        train_y_i = flux_t[i, train_mask[i]]
        
        if train_x_i.shape[0] < 2:
            mean_val = float(train_y_i.mean().item()) if train_y_i.numel() > 0 else float(flux_t[i].mean().item())
            drift[i, :] = torch.full((T,), mean_val, device=device, dtype=dtype)
            continue
            
        # Инициализируем likelihood и модель
        likelihood = GaussianLikelihood().to(device)
        model = FluxDriftGP(train_x_i.to(device=device, dtype=dtype), 
                            train_y_i.to(device=device, dtype=dtype), 
                            likelihood).to(device)
        
        # Переводим модель в режим обучения
        model.train()
        likelihood.train()
        
        # Используем LBFGS оптимизатор
        # optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=1e-2)
        optimizer = torch.optim.LBFGS(list(model.parameters()) + list(likelihood.parameters()),
                                      max_iter=20, line_search_fn='strong_wolfe')
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Сохраняем резервную копию параметров
        state_model_before = copy.deepcopy(model.state_dict())
        state_lik_before = copy.deepcopy(likelihood.state_dict())

        try:
            # Нормальный цикл обучения (несколько итераций Adam)
            for _step in range(200):          # при необходимости уменьшите/увеличьте число шагов
                optimizer.zero_grad()
                output = model(train_x_i)
                loss = -mll(output, train_y_i)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss during GP training")
                loss.backward()
                optimizer.step()

        except Exception as e:
            # Ошибка при обучении GP — откатываем параметры и используем запасной план
            warnings.warn(f"GP training failed for batch element {i}: {repr(e)}. Using Gaussian smoothing fallback.")
            try:
                model.load_state_dict(state_model_before)
                likelihood.load_state_dict(state_lik_before)
            except Exception:
                # если откат невозможен — пропускаем
                pass

            # --- Запасной план: гауссово сглаживание (конволюция) ---
            # Соберём полную кривую, но заменим точки в транзите на линейную интерполяцию из соседних
            y_full = flux_t[i].clone().to(device=device, dtype=dtype)

            # Простое локальное гауссово сглаживание с kernel_radius (в пикселях)
            kernel_radius = max(1, T // 100)   # регулируйте размер ядра
            sigma = kernel_radius / 2.0
            # формируем 1D Gaussian kernel
            coords = torch.arange(-kernel_radius, kernel_radius+1, device=device, dtype=dtype)
            gauss_kernel = torch.exp(-(coords**2) / (2 * sigma**2))
            gauss_kernel = gauss_kernel / gauss_kernel.sum()
            gauss_kernel = gauss_kernel.view(1, 1, -1)  # для F.conv1d

            # подготовка: [N=1, C=1, L=T]
            y_pad = y_full.unsqueeze(0).unsqueeze(0)
            # padding = kernel_radius на обе стороны
            y_smooth = F.conv1d(F.pad(y_pad, (kernel_radius, kernel_radius), mode='reflect'), gauss_kernel)
            y_smooth = y_smooth.view(-1)  # длина T

            # В зоне транзита возьмём сглаженное значение, вне транзита оставим оригинал
            l, r = bounds[i]
            li = int(max(0, np.floor(l)))
            ri = int(min(T, np.ceil(r)))
            if ri > li:
                y_fallback = y_full.clone()
                y_fallback[li:ri] = y_smooth[li:ri]
            else:
                y_fallback = y_smooth

            # Записываем fallback в drift (получаем константу/предсказание)
            # Чтобы сохранить семантику drift как "предсказанный тренд", используем y_fallback
            drift[i, :] = y_fallback

        else:
            # Если обучение прошло успешно — делаем предсказание
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x_i = x[i].reshape(-1, 1).to(device=device, dtype=dtype)
                observed_pred = likelihood(model(test_x_i))
                pred_mean = observed_pred.mean
                drift[i, :] = pred_mean
    
    # Нормализуем drift
    means = drift.mean(dim=1, keepdim=True)
    eps = torch.finfo(dtype).eps
    means = torch.where(means.abs() < eps, torch.full_like(means, eps), means)
    drift_ratio = drift / means

    return drift_ratio

def make_smooth_cap_gpytorch(flux_t: torch.Tensor) -> torch.Tensor:
    B, T = flux_t.shape
    device = flux_t.device
    dtype = flux_t.dtype

    x = torch.linspace(0, 1, T, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
    flux_drift = get_flux_drift_ratio_gpytorch(flux_t)  # (B, T)
    flux_smooth = torch.zeros_like(flux_t)

    base_jitter = 1e-6 if dtype == torch.float32 else 1e-12
    initial_noise = 1e-2
    min_noise = 1e-8
    max_noise = 1.0

    for i in range(B):
        train_x_i = x[i].reshape(-1, 1)
        train_y_i = flux_t[i].to(dtype).clone()

        # quick fallbacks
        if T < 6 or torch.allclose(train_y_i, train_y_i[0], atol=1e-8):
            flux_smooth[i] = train_y_i
            continue

        # normalize y
        y_mean = float(train_y_i.mean().item())
        y_std = float(train_y_i.std().item())
        y_std = max(y_std, 1e-6)
        train_y_norm = (train_y_i - y_mean) / y_std

        # Большой try/except: любая ошибка -> fallback smoothing
        try:
            likelihood = GaussianLikelihood().to(device=device, dtype=dtype)
            likelihood.noise = torch.tensor(initial_noise, device=device, dtype=dtype)

            model = FluxDriftGP(train_x_i, train_y_norm, likelihood).to(device=device, dtype=dtype)

            # clamp params if exist
            try:
                for name, param in model.named_parameters():
                    if "lengthscale" in name:
                        param.data.clamp_(1e-3, 10.0)
                    if "raw_outputscale" in name or "outputscale" in name:
                        param.data.clamp_(1e-6, 10.0)
            except Exception:
                pass

            model.train()
            likelihood.train()

            params = list(model.parameters()) + [p for p in likelihood.parameters() if id(p) not in {id(x) for x in model.parameters()}]
            optimizer = torch.optim.Adam(params, lr=0.05)
            mll = ExactMarginalLogLikelihood(likelihood, model)

            max_iter = 80
            for _ in range(max_iter):
                optimizer.zero_grad()
                output = model(train_x_i)
                loss = -mll(output, train_y_norm)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                with torch.no_grad():
                    if hasattr(likelihood, "noise"):
                        likelihood.noise.clamp_(min_noise, max_noise)
                if loss.item() < 1e-6:
                    break

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(max(base_jitter, 1e-4)):
                observed_pred = likelihood(model(train_x_i))
                mean_norm = observed_pred.mean
                if torch.isnan(mean_norm).any() or torch.isinf(mean_norm).any():
                    raise RuntimeError("Invalid GP mean")
                mean = mean_norm * y_std + y_mean
                flux_smooth[i] = mean

        except Exception:
            # единый fallback: скользящее среднее (окно адаптивного размера)
            try:
                k = max(3, min(11, (T // 10) | 1))  # нечетное окно 3..11
                pad = k // 2
                padded = torch.nn.functional.pad(train_y_i.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect').squeeze()
                sm = torch.nn.functional.avg_pool1d(padded.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1).squeeze()
                if sm.numel() == T:
                    flux_smooth[i] = sm
                else:
                    flux_smooth[i] = torch.nn.functional.interpolate(
                        sm.unsqueeze(0).unsqueeze(0), size=T, mode='linear', align_corners=False
                    ).squeeze()
            except Exception:
                flux_smooth[i] = train_y_i

    ratio = flux_smooth / (flux_drift + 1e-12)
    ratio = torch.clamp(ratio, 0.0, 1.0)

    bounds = [_get_transit_bounds_torch(flux_t[i]) for i in range(B)]
    for i in range(B):
        external_l, external_r = bounds[i]
        TT = ratio.shape[1]
        if external_l > 0:
            ratio[i, :external_l] = 1.0
        if external_r < TT:
            ratio[i, external_r:] = 1.0

        internal_l, internal_r = bounds[i]
        if external_l < internal_l and (internal_l - external_l) > 3:
            left_idx = slice(external_l, internal_l)
            y_left = ratio[i, left_idx]
            if y_left.numel() > 0:
                ratio[i, left_idx] = torch.linspace(1.0, float(y_left[-1].item()), len(y_left), device=device, dtype=dtype)
        if internal_r < external_r and (external_r - internal_r) > 3:
            right_idx = slice(internal_r, external_r)
            y_right = ratio[i, right_idx]
            if y_right.numel() > 0:
                ratio[i, right_idx] = torch.linspace(float(y_right[0].item()), 1.0, len(y_right), device=device, dtype=dtype)

    return ratio

# Ускоренный FFT ресэмплер с поддержкой батчей
def resample_fft_torch(x: torch.Tensor, new_len: int) -> torch.Tensor:
    orig_shape = x.shape
    old_len = orig_shape[-1]
    if new_len == old_len:
        return x
    
    Xf = torch.fft.rfft(x, dim=-1)
    new_bins = new_len // 2 + 1
    
    # Эффективное дополнение нулями
    if Xf.shape[-1] < new_bins:
        pad_size = new_bins - Xf.shape[-1]
        Xf = F.pad(Xf, (0, pad_size))
    else:
        Xf = Xf[..., :new_bins]
    
    Xf = Xf * (new_len / old_len)
    return torch.fft.irfft(Xf, n=new_len, dim=-1)

# Батчевая версия функций, которые остаются в NumPy
def _process_batch_numpy(func, tensor_batch):
    """Вспомогательная функция для обработки батча через numpy функции"""
    np_batch = tensor_batch.detach().cpu().numpy()
    results = []
    for i in range(np_batch.shape[0]):
        results.append(func(np_batch[i]))
    return torch.tensor(np.array(results), dtype=tensor_batch.dtype, device=tensor_batch.device)

# Батчевые версии numpy функций
def _stellar_spectrum_batch(cds_signal_t: torch.Tensor) -> torch.Tensor:
    return _process_batch_numpy(get_stellar_spectrum, cds_signal_t)

def _flux_drift_batch(flux_t: torch.Tensor) -> torch.Tensor:
    return _process_batch_numpy(get_flux_drift_ratio, flux_t)

def _make_smoothed_cap_batch(flux_t: torch.Tensor) -> torch.Tensor:
    return _process_batch_numpy(make_smooth_cap, flux_t)

def _get_transit_bounds_batch(flux_t: torch.Tensor) -> List[Tuple[int, int]]:
    np_batch = flux_t.detach().cpu().numpy()
    results = []
    for i in range(np_batch.shape[0]):
        results.append(get_transit_bounds_internal(np_batch[i]))
    return results

# Основные батчевые реализации
def _get_noise_batch(cds_signal_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    B, t_len, y_len, l_len = cds_signal_t.shape
    
    # Вычисление flux для всего батча
    flux = cds_signal_t.mean(dim=(2, 3))  # (B, t)
    
    # Векторизованные вычисления
    smoothed_cap = make_smooth_cap_gpytorch(flux)  # (B, t)
    m = smoothed_cap.min(dim=1, keepdim=True)[0]  # (B, 1)
    M = smoothed_cap.max(dim=1, keepdim=True)[0]  # (B, 1)
    transit_ratio = 1.0 - (smoothed_cap - m) / (M - m + 1e-12)  # (B, t)

    # Обработка target
    if target_t.dim() == 1:
        target_t = target_t.unsqueeze(0).expand(B, -1)
    
    # Ресэмплинг для всего батча
    target_resampled = resample_fft_torch(target_t[:, 1:], l_len)  # (B, l_len)

    # Векторизованные внешние произведения
    transit_spectrum = 1.0 - transit_ratio.unsqueeze(2) * target_resampled.unsqueeze(1)  # (B, t, l)

    # Stellar spectrum для батча
    stellar = _stellar_spectrum_batch(cds_signal_t)  # (B, l)
    stellar_spectrum = stellar.unsqueeze(1).expand(-1, t_len, -1)  # (B, t, l)

    clear_spectrum = transit_spectrum * stellar_spectrum  # (B, t, l)

    # Drift для батча
    drift = get_flux_drift_ratio_gpytorch(flux)  # (B, t)
    drift_2d = drift.unsqueeze(2).expand(-1, -1, l_len)  # (B, t, l)
    
    noise = cds_signal_t.mean(dim=2) - drift_2d * clear_spectrum  # (B, t, l)
    return noise

def _get_target_batch(cds_signal_t: torch.Tensor, noise_t: torch.Tensor) -> torch.Tensor:
    B, t_len, y_len, l_len = cds_signal_t.shape
    
    # Вычисляем flux для всего батча
    flux = cds_signal_t.mean(dim=(2, 3))  # (B, t)
    
    # Используем GPyTorch версии функций
    smoothed_cap = make_smooth_cap_gpytorch(flux)  # (B, t)
    drift = get_flux_drift_ratio_gpytorch(flux)  # (B, t)
    
    # Векторизованные вычисления статистик
    m = smoothed_cap.min(dim=1, keepdim=True)[0]  # (B, 1)
    M = smoothed_cap.max(dim=1, keepdim=True)[0]  # (B, 1)
    transit_ratio = 1.0 - (smoothed_cap - m) / (M - m + 1e-12)  # (B, t)

    # Вычисляем spectrum
    spectrum = cds_signal_t.mean(dim=2) - noise_t  # (B, t, l)
    drift_2d = drift.unsqueeze(2).expand(-1, -1, l_len)  # (B, t, l)
    clear_spectrum = spectrum / (drift_2d.abs() + 1e-8)  # (B, t, l)

    # Stellar spectrum для батча
    stellar = _stellar_spectrum_batch(cds_signal_t)  # (B, l)
    stellar_spectrum = stellar.unsqueeze(1).expand(-1, t_len, -1)  # (B, t, l)

    # Вычисляем transit spectrum
    transit_spectrum = 1.0 - (clear_spectrum / stellar_spectrum).clamp(0.0, 1.0)  # (B, t, l)
    transit_spectrum_2d = transit_ratio.unsqueeze(2).expand(-1, -1, l_len)  # (B, t, l)

    # Получаем границы транзита для всего батча
    bounds = _get_transit_bounds_batch(flux)  # Список кортежей (l_idx, r_idx)
    
    # Создаем маску для областей транзита
    transit_mask = torch.zeros(B, t_len, dtype=torch.bool, device=cds_signal_t.device)
    for i, (l_idx, r_idx) in enumerate(bounds):
        transit_mask[i, l_idx:r_idx] = True
    
    # Применяем маску и вычисляем среднее значение
    ratio = transit_spectrum / (transit_spectrum_2d + 1e-12)
    
    # Используем маску для выбора только областей транзита
    # Для этого нам нужно расширить маску до размерности (B, t, l)
    transit_mask_3d = transit_mask.unsqueeze(2).expand(-1, -1, l_len)
    
    # Применяем маску и вычисляем среднее по временной оси
    # Сначала устанавливаем значения вне транзита в ноль
    ratio_masked = ratio * transit_mask_3d.float()
    
    # Вычисляем сумму и количество элементов для каждого канала
    sum_ratio = ratio_masked.sum(dim=1)  # (B, l)
    count_ratio = transit_mask_3d.sum(dim=1).float()  # (B, l)
    
    # Вычисляем среднее значение, избегая деления на ноль
    target = sum_ratio / (count_ratio + 1e-12)  # (B, l)
    
    return target

# Обновленные основные функции
def get_noise(cds_signal, target):
    if not isinstance(cds_signal, torch.Tensor):
        # NumPy путь
        return _get_noise_numpy_single(np.asarray(cds_signal), np.asarray(target))
    
    # Torch путь
    if cds_signal.dim() == 4:
        return _get_noise_batch(cds_signal, target)
    else:
        return _get_noise_batch(cds_signal.unsqueeze(0), target).squeeze(0)

def get_target(cds_signal, noise=0):
    if not isinstance(cds_signal, torch.Tensor):
        # NumPy путь
        return _get_target_numpy_single(np.asarray(cds_signal), np.asarray(noise))
    
    # Torch путь
    if cds_signal.dim() == 4:
        return _get_target_batch(cds_signal, noise)
    else:
        return _get_target_batch(cds_signal.unsqueeze(0), noise).squeeze(0)
