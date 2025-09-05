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


def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return np.array(img_grayscale)

inspections = os.listdir("../data/")
inspections.remove(".gitkeep")
inspections.remove("PIPE_IMG.jpg")
inspections = [inspection for inspection in inspections if not inspection.endswith(".pkl")]

col1, col2, col3, col4 = [], [], [], []
header = ["inspection", "filename", "order", "img"]

data = {
    "inspection": [],
    "filename": [],
    "order": [],
    "img": [],
    "delta_x": [],
    "delta_z": [],
}

for inspection in inspections:
    imgs_path = "../data/" + inspection + "/"
    imgs_name = os.listdir(imgs_path)
    imgs_name = [name for name in imgs_name if not name.endswith(".csv")]
    imgs_name.sort()

    data['order'].extend(np.array(range(len(imgs_name))))
    data['filename'].extend(imgs_name)
    data['inspection'].extend([inspection] * len(imgs_name))
    data['img'].extend([load_img(imgs_path + img_name) for img_name in imgs_name])
    data['delta_x'].extend(len(imgs_name) * [None])
    data['delta_z'].extend(len(imgs_name) * [None])

dataset_df = pd.DataFrame(data)
dataset_df.to_pickle("../data/dataset.pkl")