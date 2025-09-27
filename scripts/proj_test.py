import numpy as np
import pandas as pd
import time
from visual_odometer import VisualOdometer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse.linalg import svds as svds_cpu
DATA_ROOT = "../data/"

def add_gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise

data = pd.read_pickle(DATA_ROOT + "dataset.pkl")


odometer = VisualOdometer(img_shape=data["img"][0].shape, xres=-2, yres=-2)
# odometer.config_displacement_estimation("phase-correlation")
odometer.config_displacement_estimation("svd")

odometer.feed_image(data["img"][0])
odometer.get_displacement()
odometer.feed_image(data["img"][1])
odometer.get_displacement()


imgs = [data.iloc[1000]['img'], data.iloc[1015]['img']]

# imgs[0] = add_gaussian_noise(imgs[0], sigma=50)
# imgs[1] = add_gaussian_noise(imgs[1], sigma=50)

imgs_spectra = [np.fft.fft2(img) for img in imgs]


Cn = (np.conj(imgs_spectra[0]) * imgs_spectra[1]) / (np.abs(np.conj(imgs_spectra[0]) * imgs_spectra[1]) + 1E-9)
Cn_t = np.real(np.fft.ifft2(Cn))

t0 = time.time()
qu_svd, s, qv_svd = svds_cpu(Cn, k=1)
Cn_SVD = np.fft.fftshift(qu_svd @ np.diag(s) @ qv_svd)
t_svd = time.time() - t0

#%%

t0 = time.time()
R = 6
dx_min, dx_max = 0, 50
dy_min, dy_max = 0, 50

Cn_t_proj = Cn_t.copy()
mask = np.zeros_like(Cn_t_proj)
# Top-left
mask[dy_min:int(dy_max + R/2), dx_min:int(dx_max + R/2)] = 1.

# Top-right
mask[dy_min:int(dy_max + R/2), int(Cn_t.shape[1] - dx_max - R/2):Cn_t.shape[1] - dx_min] = 1.

# Bottom-left
mask[int(Cn_t.shape[0] - dy_max - R/2):Cn_t.shape[0] - dy_min, dx_min:int(dx_max + R/2)] = 1.

# Bottom-right
mask[int(Cn_t.shape[0] - dy_max - R/2):Cn_t.shape[0] - dy_min,
     int(Cn_t.shape[1] - dx_max - R/2):Cn_t.shape[1] - dx_min] = 1.

Cn_t_proj = Cn_t_proj * mask
Cn_proj = np.fft.fft2(Cn_t_proj)

qu_svd_proj, s, qv_svd_proj = svds_cpu(Cn_proj, k=1)
Cn_proj_svd = np.fft.fftshift(qu_svd_proj @ np.diag(s) @ qv_svd_proj)
t_svd_proj = time.time() - t0

#%% PCA based method:import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Compute phase correlation
Cn = (np.conj(imgs_spectra[0]) * imgs_spectra[1]) / (np.abs(np.conj(imgs_spectra[0]) * imgs_spectra[1]) + 1e-9)

# Step 2: Compute wrapped phase
Q = np.angle(np.fft.fftshift(Cn))

# Step 3: Image shape
h, w = Q.shape

# Step 4: Pixel coordinates
Y, X = np.mgrid[0:h, 0:w]
coords = np.column_stack((X.ravel(), Y.ravel()))  # (n_pixels, 2)

# Step 5: Phase magnitude as weights
weights = np.abs(Q).ravel()

# Step 6: Center coordinates using weighted mean
coords_centered = coords - np.average(coords, axis=0, weights=weights)

# Step 7: Apply PCA with weights
coords_weighted = coords_centered * np.sqrt(weights[:, None])
pca = PCA(n_components=2)
pca.fit(coords_weighted)

# Step 8: Project coordinates onto PCA axes
proj = pca.transform(coords_centered)  # shape (n_pixels, 2)
proj_PC1 = proj[:, 0].reshape(h, w)
proj_PC2 = proj[:, 1].reshape(h, w)

# Step 9: Plot original Q and projections
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Original phase
axs[0].imshow(Q, cmap='gray', origin='upper')
axs[0].set_title('Original Phase (Q)')

# Projection along PC1
im1 = axs[1].imshow(proj_PC1, cmap='jet', origin='upper')
axs[1].set_title('Projection along PC1')
fig.colorbar(im1, ax=axs[1])

# Projection along PC2
im2 = axs[2].imshow(proj_PC2, cmap='jet', origin='upper')
axs[2].set_title('Projection along PC2')
fig.colorbar(im2, ax=axs[2])

plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for img, ax in zip(imgs, axs):
    ax.imshow(img, cmap='gray', aspect='equal', interpolation="None")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

#%%

fig, axs = plt.subplots(3, 3, figsize=(14, 8))

methods = ["SVD", "Projection + SVD"]
elapsed_time = [t_svd, t_svd_proj]
qu = [qu_svd, qu_svd_proj]
qv = [qv_svd, qv_svd_proj]

for i, Q in enumerate([Cn_SVD, Cn_proj_svd]):


    axs[0, i].imshow(np.angle(Q), cmap='gray')
    axs[0, i].set_title(f"{methods[i]}.\n Elapsed time: {elapsed_time[i] * 1e3:.2f} ms")
    axs[1, i].plot(np.fft.fftshift(np.angle(qu[i])[:, 0]), '-o', linewidth=1.5, markersize=2,  color='y')
    axs[2, i].plot(np.fft.fftshift(np.angle(qv[i])[0, :]), '-o', linewidth=1.5, markersize=2,  color='b')
plt.show()