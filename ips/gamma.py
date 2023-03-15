import numpy as np


def apply_gamma_base(x):
    return x ** (1.0 / 2.2)


def apply_gamma_orj(x):
    x = x.copy()
    idx = x <= 0.0031308
    x[idx] *= 12.92
    x[idx == False] = (x[idx == False] ** (1.0 / 2.4)) * 1.055 - 0.055
    return x


def apply_gamma_channel_wise(x):
    cx = x.copy()
    h, w, c = cx.shape
    rx = cx.reshape((h * w, c))
    for i in range(c):
        values, _ = np.histogram(rx[:, i], bins=128)
        cutoff = np.ceil((values.cumsum() / (h * w))[0] * 100).astype(int)
        low, high = np.percentile(rx[:, i], [cutoff, 100])
        idx = (rx[:, i] <= low) & (rx[:, i] >= high)
        rx[idx, i] *= 12.92
        rx[idx == False, i] = (rx[idx == False, i] ** (1.0 / 2.4)) * 1.055 - 0.055
    return rx.reshape((h, w, c))

