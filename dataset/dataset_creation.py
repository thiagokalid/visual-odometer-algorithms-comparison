"""
O movimento realizado pelo encoder foi aproximadamente:
1. 110 mm na horizontal para a direita ➡️
2. 120 mm na vertical para baixo ⬇️

Para conversão de pixels para mm a taxa correta é de 22.2 px/mm.
"""

import numpy as np
import pandas as pd
import os
from PIL import Image, ImageOps
import time

IMG_FILE_EXTENSION = (".png", ".jpg")
COLD_RUN = True  # Reprocess everything regardless if already exist a dataset.pkl

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return np.array(img_grayscale)

data_root = "../dataset/"
inspections = [
    name for name in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, name))
]


col1, col2, col3, col4 = [], [], [], []
header = ["inspection", "filename", "order", "img"]



# Read images:
t0 = time.time()
for i, inspection in enumerate(inspections):
    data = {
        "inspection": [],
        "filename": [],
        "order": [],
        "img": [],
        "delta_x": [],
        "delta_y": []
    }


    imgs_path = data_root + inspection
    dataset_pkl_exist = os.path.exists(imgs_path)

    if (not COLD_RUN) and dataset_pkl_exist:
        pass
    else:
        imgs_name = os.listdir(imgs_path)
        imgs_name = [f for f in imgs_name if f.lower().endswith(IMG_FILE_EXTENSION)]
        imgs_name.sort()

        # Read CSV dataset:
        try:
            calibration_data = pd.read_csv(imgs_path + "/calibration_data.csv")
            px_per_mm = np.float32(calibration_data["px_p_mm"])
        except FileNotFoundError:
            px_per_mm = 1

        try:
            expected_path = pd.read_csv(imgs_path + "/true_positions.csv")
        except FileNotFoundError:
            expected_path = np.array(len(imgs_name) * [0.0])

        data['order'].extend(np.array(range(len(imgs_name))))
        data['filename'].extend(imgs_name)
        data['inspection'].extend([inspection] * len(imgs_name))
        data['img'].extend([load_img(imgs_path + "/" + img_name) for img_name in imgs_name])
        data['delta_x'].extend(px_per_mm * expected_path)
        data['delta_y'].extend(px_per_mm * expected_path)

        dataset_df = pd.DataFrame(data)
        dataset_df.to_pickle(imgs_path + "/dataset.pkl")

    print(
        f"Dataset processing progress: {(i + 1)}/{len(inspections)} "
        f"({(i + 1) / len(inspections):.2%}) | "
        f"Elapsed time: {time.time() - t0:.1f} s",
        end="\r"
    )