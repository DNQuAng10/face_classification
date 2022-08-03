import time

from OpenVinoModel import OpenVinoModel
import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
import tqdm

CWD = pathlib.Path(__file__).resolve().parent


DIR = "/mnt/datadrive/quangdn/far/face_classification/data/sub_eval"
DIR = "/home/quangdn/far/data/extract_face_from_frame"
DIR = "/home/quangdn/far/data/extract_face_from_image"
SAVE = "classify_androids_video"
SAVE = "classify_eq_data"

predict_cases = np.zeros([3,3])

times = []

INPUT_SIZE = (112, 112)

ovn_model = "/home/quangdn/far/face_classification/models/112_Classify_Adam_Epoch_75_Batch_6750_95.657_97.667_Time_1634623345.5846994_checkpoint.xml"
classify = OpenVinoModel(ovn_model, input_size=INPUT_SIZE)
print("Loading model Done...")

_class = ["glasses", "mask", "normal"]
_count = dict()
_count["glasses"] = 0
_count["mask"] = 0
_count["normal"] = 0

for path, dirs, files in os.walk(DIR):
    for filename in tqdm.tqdm(files):
        if pathlib.Path(filename).suffix not in [".jpg", ".png"]:
            continue
        img = cv2.imread(os.path.join(path, filename))
        output = np.array(classify.predict(img))
        output = (np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
        output_dir = os.path.join(CWD, SAVE, _class[np.argmax(output)])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _count[_class[np.argmax(output)]] += 1
        cv2.imwrite(os.path.join(output_dir, filename), img)
print(_count)

