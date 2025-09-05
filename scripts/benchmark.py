import numpy as np
import pandas as pd
import time
from visual_odometer import VisualOdometer
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_ROOT = "../data/"

data = pd.read_pickle(DATA_ROOT + "artificial_dataset.pkl")
results = {
    "mean elapsed time / (ms)": [],
    "max elapsed time / (ms)": [],
    "min elapsed time / (ms)": [],
    "error_x / (mm)": [],
    "error_y / (mm)": [],
    "error_norm / (mm)": [],
}


odometer = VisualOdometer(img_shape=data["img"][0].shape, xres=-1, yres=-1)
odometer.config_displacement_estimation("phase-correlation")
# odometer.config_displacement_estimation("svd")

odometer.feed_image(data["img"][0])
odometer.get_displacement()
odometer.feed_image(data["img"][1])
odometer.get_displacement()

t0 = time.time()
elapsed_time = []

nrows = len(data)
delta_x_pred, delta_z_pred = [], []
x_pred, z_pred = [0], [0]
x_true, z_true = [0], [0]
for i in tqdm(range(nrows)):
    order = data.iloc[i]['order']
    img = data.iloc[i]['img']


    ti0 = time.time()
    odometer.feed_image(img)
    delta_xi, delta_zi = odometer.get_displacement()
    ti1 = time.time()

    elapsed_time.append((ti1 - ti0) * 1e3)
    delta_x_pred.append(delta_xi)
    delta_z_pred.append(delta_zi)
    x_pred.append(delta_xi + x_pred[-1])
    z_pred.append(delta_zi + z_pred[-1])
    x_true.append(data["delta_x"][i] + x_true[-1])
    z_true.append(data["delta_z"][i] + z_true[-1])

x_true, z_true = np.array(x_true), np.array(z_true)
x_pred, z_pred = np.array(x_pred), np.array(z_pred)

#%% Qualitative values:
import matplotlib
matplotlib.use("TkAgg")

plt.figure(figsize=(12, 8))
plt.plot(x_pred, z_pred, '-o', color='blue', label="Estimated (Hoge)")
plt.plot(x_true, z_true, '-o', color='r', alpha=.2, label="Ground truth")
plt.legend()
plt.xlabel("x-axis / (pixels)")
plt.ylabel("y-axis / (pixels)")

#%%

error = np.sqrt((x_true - x_pred)**2 + (z_true - z_pred)**2)
print(f"Mean error x-axis = {np.mean(np.abs(x_pred - x_true)):.2f} +- {np.std(np.abs(x_pred - x_true)):.2f}")
print(f"Mean error z-axis = {np.mean(np.abs(z_pred - z_true)):.2f} +- {np.std(np.abs(z_pred - z_true)):.2f}")
print(f"Mean error = {np.mean(error):.2f} +- {np.std(error):.2f}")

cum_error_x, cum_error_y = x_pred[-1] - x_true[-1], z_pred[-1] - z_true[-1]
results["error_x / (mm)"] = cum_error_x
results["error_y / (mm)"] = cum_error_y
results["error_norm / (mm)"] = np.sqrt(cum_error_x**2 + cum_error_y**2)

Nbins = np.sqrt(len(x_true)).astype(int)

fig, ax = plt.subplots(3, 1, sharey=True)
ax[0].boxplot(x_pred - x_true)
ax[0].set_title("Error along x-axis")
ax[1].boxplot(z_pred - z_true)
ax[1].set_title("Error along z-axis")
ax[2].boxplot(error)
ax[2].set_title("Distance between true and predicted positions")
plt.show()
#%% Plot results:
plt.figure(figsize=(10, 6))
plt.plot(np.array(elapsed_time), color='k')

colors = ['lime', 'red', 'blue']
for i, estimator in enumerate([np.mean, np.max, np.min]):
    operator_name = estimator.__name__
    plt.plot(np.arange(len(elapsed_time)),estimator(elapsed_time) * np.ones_like(elapsed_time), '--', color=colors[i],
             label=f"{operator_name} = {estimator(elapsed_time):.2f} ms")

    results[operator_name + " elapsed time / (ms)"].append(estimator(elapsed_time))

plt.grid(alpha=.5)
plt.ylabel("Elapsed time per frame / (ms)")
plt.xlabel("Frame")
plt.legend()
plt.show()

#%%
results_df = pd.DataFrame(results)
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)       # don't wrap to next line
pd.set_option("display.max_colwidth", None)  # show full column content
print(results_df.round(4))
