import numpy as np
from numpy import ndarray

from scipy.signal import convolve
from scipy.sparse.linalg import svds as svds_cpu


def normalize_product(F, G):
    Q = F * np.conj(G)
    Q /= np.abs(Q)
    return Q

def phase_fringe_filter(cross_power_spectrum: ndarray, window_size: tuple = (5, 5), threshold: float = 0.03) -> ndarray:
    # Aplica o filtro de média para reduzir o ruído
    filtered_spectrum = convolve(cross_power_spectrum, np.ones(window_size) / np.prod(window_size), mode='same')  # Alterado de 'constant' para 'same'

    # Calcula a diferença entre o espectro original e o filtrado
    diff_spectrum = cross_power_spectrum - filtered_spectrum

    # Aplica o limiar para identificar as regiões de pico
    peak_mask = np.abs(diff_spectrum) > threshold

    # Atenua as regiões de pico no espectro original
    phase_filtered_spectrum = cross_power_spectrum.copy()
    phase_filtered_spectrum[peak_mask] *= 0.5  # Reduz a amplitude nas regiões de pico

    return phase_filtered_spectrum


def linear_regression(x: ndarray, y: ndarray) -> (float, float):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    x_sol = np.linalg.lstsq(R, y)
    mu, c = x_sol[0]
    return mu, c


def phase_unwrapping(phase_vec: ndarray, factor: float = 0.7) -> ndarray:
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
            phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)


def svd_estimate_shift(phase_vec: ndarray, N: int, phase_windowing=None) -> float:

    phase_unwrapped = np.unwrap(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2

    if phase_windowing == "central":
        x = r[M - 50:M + 50]
        y = phase_unwrapped[M - 50:M + 50]
    elif phase_windowing == "initial":
        x = r[M - 80:M - 10]
        y = phase_unwrapped[M - 80:M - 10]
    else:
        x = r
        y = phase_unwrapped

    mu, _ = linear_regression(x, y)

    return mu * N / (2 * np.pi)

def apply_projection_phase_filter(Cn_t, R, dx_min, dx_max, dy_min, dy_max) -> ndarray:
    # # Top-left
    # mask[dy_min:int(dy_max + R / 2), dx_min:int(dx_max + R / 2)] = 1.
    #
    # # Top-right
    # mask[dy_min:int(dy_max + R / 2), int(Cn_t.shape[1] - dx_max - R / 2):Cn_t.shape[1] - dx_min] = 1.
    #
    # # Bottom-left
    # mask[int(Cn_t.shape[0] - dy_max - R / 2):Cn_t.shape[0] - dy_min, dx_min:int(dx_max + R / 2)] = 1.
    #
    # # Bottom-right
    # mask[int(Cn_t.shape[0] - dy_max - R / 2):Cn_t.shape[0] - dy_min,
    # int(Cn_t.shape[1] - dx_max - R / 2):Cn_t.shape[1] - dx_min] = 1.
    M, N = Cn_t.shape

    delta_x = (dx_max - dx_min) + R
    delta_y = (dy_max - dy_min) + R

    lb_y, ub_y = np.clip(M//2 - delta_y//2, a_min=0, a_max=M), np.clip(M//2 + delta_y//2, a_min=0, a_max=M)
    lb_x, ub_x = np.clip(N//2 - delta_x//2, a_min=0, a_max=N), np.clip(N//2 + delta_x//2, a_min=0, a_max=N)

    mask = np.zeros_like(Cn_t, dtype=int)
    mask[
        lb_y : ub_y,
        lb_x : ub_x
    ] = 1

    # return Cn_t * np.fft.fftshift(mask)
    return np.fft.fftshift(np.fft.fftshift(Cn_t)[
        lb_y : ub_y,
        lb_x : ub_x
    ])

def proj_svd_method(fft_beg, fft_end, M: int, N: int, R: int=6, dx_min=0, dx_max=64, dy_min=0, dy_max=48, phase_windowing=None) -> (float, float):

    Q = normalize_product(fft_beg, fft_end)

    Cn_t = np.fft.ifft2(Q)

    Cn_t_proj = apply_projection_phase_filter(Cn_t, R, dx_min, dx_max, dy_min, dy_max)

    Q_filtered = np.fft.fft2(Cn_t_proj)

    qu, s, qv = svds_cpu(Q_filtered, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])


    M_ = M * Q_filtered.shape[0] / Q.shape[0]
    N_ = N * Q_filtered.shape[1] / Q.shape[1]

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltax = svd_estimate_shift(ang_qv, int(M_), phase_windowing)
    deltay = svd_estimate_shift(ang_qu, int(N_), phase_windowing)

    return deltax, deltay


