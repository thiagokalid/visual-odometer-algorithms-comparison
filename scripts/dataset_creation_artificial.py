import numpy as np
import cv2
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import os
import pandas as pd


# matplotlib.use('QtAgg')

def fft_shift_2d(imagem, dx, dy, janela=None):
    h, w = imagem.shape
    imagem_f = imagem.astype(np.float64)

    if janela is not None:
        imagem_f *= janela

    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)
    fase = np.exp(-2j * np.pi * (FX * dx + FY * dy))
    I_fft = fft2(imagem_f)
    I_deslocada = ifft2(I_fft * fase)
    return np.real(I_deslocada)


def add_gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise


def add_salt_and_pepper(img, amount=90 / 100, s_vs_p=0.5):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = (np.random.randint(0, img.shape[0], int(num_salt)),
              np.random.randint(0, img.shape[1], int(num_salt)))
    noisy[coords] = 255

    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = (np.random.randint(0, img.shape[0], int(num_pepper)),
              np.random.randint(0, img.shape[1], int(num_pepper)))
    noisy[coords] = 0
    return noisy


def apply_lens_blur(img, blur_kernel=(21, 21), feather=100):
    rows, cols = img.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 2, 1, -1)

    # Smooth transition between sharp and blurred areas
    mask = cv2.GaussianBlur(mask, (101, 101), feather)

    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    blended = img * mask + blurred * (1 - mask)
    return blended


def apply_full_blur(img, blur_kernel=(7, 7)):
    """
    Apply uniform Gaussian blur to the entire image.
    """
    return cv2.GaussianBlur(img, blur_kernel, 0)


# Exemplo de uso:

DATA_ROOT = "../data/pipe_imgs/"
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)
    print(f"Directory created: {DATA_ROOT}")
else:
    print(f"Directory already exists: {DATA_ROOT}")

base_img = cv2.cvtColor(cv2.imread("../data/PIPE_IMG.jpg"), cv2.COLOR_BGR2GRAY)

np.random.seed(42)
img_size = (480, 640)
downscale_factor = 2  # Reduce dimensions by half

# Square:
trajectory_description = [
    {
        "shift_x": 0, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.05, "shift_noise_y": 0.05
    },
    {
        "shift_x": 1000, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.5, "shift_noise_y": 0.1
    },
    {
        "shift_x": 0, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.05, "shift_noise_y": 0.05
    },
    {
        "shift_x": 0, "shift_y": 500,
        "N": 100, "shift_noise_x": 0.1, "shift_noise_y": 0.5
    },
    {
        "shift_x": 0, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.05, "shift_noise_y": 0.05
    },
    {
        "shift_x": -1000, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.5, "shift_noise_y": 0.1
    },
    {
        "shift_x": 0, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.05, "shift_noise_y": 0.05
    },
    {
        "shift_x": 0, "shift_y": -500,
        "N": 100, "shift_noise_x": 0.1, "shift_noise_y": 0.5
    },
    {
        "shift_x": 0, "shift_y": 0,
        "N": 100, "shift_noise_x": 0.05, "shift_noise_y": 0.05
    }
]

x0, z0 = img_size[1] // 2 + 20, img_size[0] // 2 + 20

data = {
    "inspection": [],
    "filename": [],
    "order": [],
    "img": [],
    "delta_x": [],
    "delta_y": [],
}

x, z = [x0], [z0]
true_shift_x, true_shift_y = [], []
imgs = []

total_iter = np.sum([description["N"] for description in trajectory_description])

curr_iter = 0
for description in trajectory_description:
    shift_x, shift_y = description["shift_x"], description["shift_y"]
    N = description["N"]
    delta_x, delta_y = shift_x / N, shift_y / N

    for i in range(1, N):
        rand_delta_x = np.random.normal(delta_x, description["shift_noise_x"])
        rand_delta_y = np.random.normal(delta_y, description["shift_noise_y"])

        x.append(rand_delta_x + x[-1])
        z.append(rand_delta_y + z[-1])

        true_shift_x.append(rand_delta_x)
        true_shift_y.append(rand_delta_y)

        int_x, int_z = np.round(x[-1]), np.round(z[-1])
        img_int = base_img[int(int_z - img_size[0] // 2): int(int_z + img_size[0] // 2),
                  int(int_x - img_size[1] // 2): int(int_x + img_size[1] // 2)]
        float_x, float_z = x[-1] - int_x, z[-1] - int_z
        shifted_img = fft_shift_2d(img_int, dx=float_x, dy=float_z)

        shifted_img = apply_lens_blur(shifted_img)
        shifted_img = apply_full_blur(shifted_img)
        shifted_img = add_gaussian_noise(shifted_img, sigma=15)
        # shifted_img = add_salt_and_pepper(shifted_img, amount=0.01)

        shifted_img = np.clip(shifted_img, 0, 255).astype(np.uint8)
        downsampled_img = cv2.resize(shifted_img, None, fx=1 / downscale_factor, fy=1 / downscale_factor)

        data['order'].append(i)
        data['filename'].append("pipe_img.jpg")
        data['inspection'].append("pipe_img")
        data['delta_x'].append(rand_delta_x / 2)
        data['delta_y'].append(rand_delta_y / 2)
        data['img'].append(downsampled_img)

        cv2.imwrite(DATA_ROOT + f"img_{curr_iter:05d}.png", downsampled_img)
        curr_iter += 1
        print(f"Progress {curr_iter} / {total_iter} ({curr_iter / total_iter:.2%})", end='\r')

x, z = np.array(x), np.array(z)

df = pd.DataFrame(data)
df.to_pickle(DATA_ROOT + "../" + "artificial_dataset.pkl")

plt.figure()
plt.plot(x, z, 'o')
plt.axis("equal")
plt.show()
