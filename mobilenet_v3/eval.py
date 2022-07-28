import time

from OpenVinoModel import OpenVinoModel
import os
import cv2
import glob
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tqdm

CWD = pathlib.Path(__file__).resolve().parent
# from modules.LandmarksCropper import LandmarksCroper
# from modules.TDDFA.TDDFA import TDDFA_Blob
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

def multitask_to_true_false_cases(file_path, output, dict_results: dict=None, list_pred: list=None):
    # glasses, mask, normal = 0, 1, 2
    global face_types_dict, predict_cases
    output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
    
    dict_result = {}
    dir_name = os.path.basename(os.path.dirname(file_path))
    if dir_name == "glasses":
        if output_classify[0] == 1:
            predict_cases[0][0] += 1
            pred = 0
            dict_result[file_path] = {"label": "glasses", "pred": "glasses"}
        else:
            predict_cases[0][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[0][2] += 1 if output_classify[1] != 1 else 0
            pred = 1 if output_classify[1] == 1 else 2
            dict_result[file_path] = {"label": "glasses", "pred": "mask" if output_classify[1] == 1 else "normal"}
            # print(predict_cases[0][1], predict_cases[0][2])
    elif dir_name == "mask":
        if output_classify[1] == 1:
            predict_cases[1][1] += 1
            pred = 1
            dict_result[file_path] = {"label": "mask", "pred": "mask"}
        else:
            predict_cases[1][0] += 1 if output_classify[0] == 1 else 0
            predict_cases[1][2] += 1 if output_classify[0] != 1 else 0
            pred = 0 if output_classify[0] == 1 else 2
            dict_result[file_path] = {"label": "mask", "pred": "glasses" if output_classify[0] == 1 else "normal"}
    elif dir_name == "normal":
        if output_classify[2] == 1:
            predict_cases[2][2] += 1
            pred = 2
            dict_result[file_path] = {"label": "normal", "pred": "normal"}
        else:
            predict_cases[2][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[2][0] += 1 if output_classify[1] != 1 else 0
            pred = 1 if output_classify[1] == 1 else 0
            dict_result[file_path] = {"label": "normal", "pred": "mask" if output_classify[1] == 1 else "glasses"}
    dict_results.update(dict_result)
    list_pred.append(pred)
    return dict_results


def single_task_to_false_cases(filename, output):
    global face_types_dict, predict_cases
    output_classify = (1 if output[0] > 0.5 else 0,
                       1 if output[1] > 0.5 else 0,
                       1 if output[0] <= 0.5 and output[1] <= 0.5 else 0)
    if os.path.basename(os.path.dirname(filename)) == "glasses":
        if output_classify[0] == 1:
            predict_cases[0][0] += 1
        else:
            predict_cases[0][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[0][2] += 1 if output_classify[1] != 1 else 0
    elif os.path.basename(os.path.dirname(filename)) == "mask":
        if output_classify[1] == 1:
            predict_cases[1][1] += 1
        else:
            predict_cases[1][0] += 1 if output_classify[0] == 1 else 0
            predict_cases[1][2] += 1 if output_classify[0] != 1 else 0
    elif os.path.basename(os.path.dirname(filename)) == "normal":
        if output_classify[2] == 1:
            predict_cases[2][2] += 1
        else:
            predict_cases[2][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[2][0] += 1 if output_classify[1] != 1 else 0

def cal_precision_recall(list_label, list_pred):
    pr = precision_score(list_label, list_pred, average=None)
    rc = recall_score(list_label, list_pred, average=None)

    cls_report = classification_report(list_label, list_pred)

    cm = confusion_matrix(list_label, list_pred)
    print("classification report: \n", cls_report)
    print("precision score: ", pr)
    print("recall score: ", rc)
    print("confusion matrix: \n", cm)


def save_result(dict_result: dict=None):
    assert dict_result is not None
    dict_data = {}
    for k, dict_result in dict_result.items():
        if k != "eval":
            data = pd.DataFrame(data={
                "image path": [k for k in dict_result.keys()],
                "label": [v["label"] for v in dict_result.values()],
                "pred": [v["pred"] for v in dict_result.values()]
            })
            dict_data[k] = data

    save_file = os.path.join(CWD, "save_result", "%s.xlsx" % os.path.basename(DIR))
    with pd.ExcelWriter(save_file) as writer:
        for k, data in dict_data.items():
            data.to_excel(writer, sheet_name=k)
    print("save done at: ", save_file)

face_types_dict = dict()
face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval"
DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_1"
DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_png_0"
DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_png_1"
# DIR = "/mnt/datadrive/quangdn/far/face_classification/data/test_eval_png_1"
# DIR = "/mnt/datadrive/quangdn/far/face_classification/data/new_aligned"
DIR = "/mnt/datadrive/quangdn/far/face_classification/data/sub_eval_labeled"

predict_cases = np.zeros([3, 3])

times = []

INPUT_SIZE = (112, 112)

ovn_model = "/home/quangdn/far/face_classification/models/112_Classify_Adam_Epoch_75_Batch_6750_95.657_97.667_Time_1634623345.5846994_checkpoint.xml"
classify = OpenVinoModel(ovn_model, input_size=INPUT_SIZE)
print("Loading model Done...")

dict_all_result = {}
list_all_label = []
list_all_pred = []
for subdir, dirs, files in os.walk(DIR):
    dict_subdir_result = {}
    for filename in tqdm.tqdm(files):
        if os.path.basename(subdir) == "glasses":
            label = 0
        elif os.path.basename(subdir) == "mask":
            label = 1
        elif os.path.basename(subdir) == "normal":
            label = 2
        list_all_label.append(label)
        if pathlib.Path(filename).suffix not in [".jpg", ".png"]:
            continue
        file_path = os.path.join(subdir, filename)
        face = cv2.imread(file_path)
        # face = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        # face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)
        w, h, c = face.shape
        if face is None:
            continue
        t = time.time()
        output = np.array(classify.predict(face))
        times.append(time.time() - t)
        multitask_to_true_false_cases(file_path, output, dict_subdir_result, list_all_pred)
        # single_task_to_false_cases(filename, output[0][0])
    dict_all_result[os.path.basename(subdir)] = dict_subdir_result

cal_precision_recall(list_all_label, list_all_pred)

print("AVG time:", np.array(times).mean())
df_cm = pd.DataFrame(predict_cases, columns=["Glass\npredicted", "Mask\npredicted", "Normal\npredicted"],
                     index=["Glass", "Mask", "Normal"])
fig = plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
fig.tight_layout()
# plt.show()
plt.savefig(os.path.join(CWD, "save_result", "%s.png" % os.path.basename(DIR)))

# subdir = [os.path.basename(sd) for sd in glob.glob("%s/*" % DIR) if os.path.isdir(sd)]
# print(subdir)

# print(dict_all_result.keys(), [len(v) for k, v in dict_all_result.items()])
save_result(dict_all_result)
