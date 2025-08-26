import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import itertools
from .process_transit import get_transit_bounds_external, get_transit_bounds_internal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.exceptions import ConvergenceWarning


def calibrate_signal(signal, dead, dark, flat, linear_corr=None):
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
                   flat=None, 
                   gain=None, offset=None,
                   binning=30,
                   calibrate=True) -> np.ndarray:
    def restore(signal, gain, offset):
        return signal / gain + offset

    signal = signal.astype(np.float64)
    signal = restore(signal, gain, offset)
    signal = calibrate_signal(signal, dead, dark, 
                              flat, linear_corr=linear_corr) if calibrate else signal
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

