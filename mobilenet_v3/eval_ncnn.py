import argparse
import time

import os
import cv2
import glob
import ncnn
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import pickle
import tqdm

CWD = pathlib.Path(__file__).resolve().parent
# from modules.LandmarksCropper import LandmarksCroper
# from modules.TDDFA.TDDFA import TDDFA_Blob
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report


CLASSES = ["glasses", "mask", "normal"]


def init_ncnn(ncnn_bin_path, ncnn_param_path):
    ### Run step by step lib
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True

    # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    print("Load param...")
    net.load_param(ncnn_param_path)
    print("Load bin...")
    net.load_model(ncnn_bin_path)
    print("Net: ", net)
    return net

def ncnn_infer(test_input, net):
    mean_vals = [127.5, 127.5, 127.5]
    # norm_vals = [1 / 128.0, 1 / 128.0, 1 / 128.0]
    norm_vals = [1 / 127.5, 1 / 127.5, 1 / 127.5]
    # mean_vals = []
    # norm_vals = []
    
    num_threads = 4
    mat_in = ncnn.Mat.from_pixels_resize(
        test_input,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        test_input.shape[1], test_input.shape[0],
        target_size,
        target_size,
    )
    # mat_in = ncnn.Mat(test_input)
    # input_diff = mat_in - test_input
    # print("test_input: ", test_input)
    # print("mat in: ", mat_in)
    # print("input different: ", input_diff, input_diff.shape)
    mat_in.substract_mean_normalize(mean_vals, norm_vals)
    # print("mat_in.shape: ", mat_in)

    ex = net.create_extractor()
    ex.set_num_threads(num_threads)

    ex.input("input", mat_in)

    ret0, mat_out0 = ex.extract("output0")
    ret1, mat_out1 = ex.extract("output1")
    out0 = np.array(mat_out0).reshape((1, 2))
    out1 = np.array(mat_out1).reshape((1, 2))
    output = np.array([out0, out1])
    return output

def multitask_to_true_false_cases(file_path, output, dict_results: dict=None, list_pred: list=None):
    # glasses, mask, normal = 0, 1, 2
    global face_types_dict, predict_cases
    output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
    # print("output_classify: ", output_classify)

    dict_result = {}
    dir_name = os.path.basename(os.path.dirname(file_path))
    # print("==. dir_name: ", dir_name)
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

def cal_accuracy_params_for_cm(cm, is_round=True):
    shape, shape = cm.shape
    dict_tp = {}
    dict_fp = {}
    dict_tn = {}
    dict_fn = {}
    for i in range(shape):
        for j in range(shape):
            if i == j:
                dict_tp[i] = cm[i][j]
                dict_fp[i] = sum([cm[r][j] for r in range(shape) if r != i])
                dict_fn[i] = sum([cm[i][c] for c in range(shape) if c != i])
                dict_tn[i] = sum([cm[rc][rc] for rc in range(shape) if rc != i])

    dict_precision = {}
    dict_recall = {}
    dict_fpr = {}
    dict_fnr = {}
    for i in range(shape):
        if is_round:
            dict_precision[i] = round(dict_tp[i] / (dict_tp[i] + dict_fp[i]), 2)
            dict_recall[i] = round(dict_tp[i] / (dict_tp[i] + dict_fn[i]), 2)
            dict_fpr[i] = round((dict_fp[i] / (dict_fp[i] + dict_tn[i])) * 100, 2)
            dict_fnr[i] = (1 - dict_recall[i]) * 100
        else:
            dict_precision[i] = dict_tp[i] / (dict_tp[i] + dict_fp[i])
            dict_recall[i] = dict_tp[i] / (dict_tp[i] + dict_fn[i])
            dict_fpr[i] = (dict_fp[i] / (dict_fp[i] + dict_tn[i])) * 100
            dict_fnr[i] = (1 - dict_recall[i]) * 100
    # dict_ = map(lambda x: x.update({"avr": sum(x.values())}), [dict_tp, dict_fp, dict_tn, dict_fn, dict_precision, dict_recall, dict_fpr])
    for dict_ in [dict_tp, dict_fp, dict_tn, dict_fn, dict_precision, dict_recall, dict_fpr, dict_fnr]:
        dict_ = dict_.update({"avr": round(sum(dict_.values()) / shape, 2)})

    print("TP: ", dict_tp)
    print("FP: ", dict_fp)
    print("TN: ", dict_tn)
    print("FN: ", dict_fn)
    print("Precision: ", dict_precision)
    print("Recall: ", dict_recall)
    print("FPR: ", dict_fpr)
    print("FNR: ", dict_fnr)

    # classes = [CLASSES[i] for i in dict_tp.keys()]
    data = pd.DataFrame(data={
        "classes": ["glasses", "mask", "normal", "AVR"],
        "TP": [i for i in dict_tp.values()],
        "FP": [i for i in dict_fp.values()],
        "TN": [i for i in dict_tn.values()],
        "FN": [i for i in dict_fn.values()],
        "precision": [i for i in dict_precision.values()],
        "recall": [i for i in dict_recall.values()],
        "FPR": [i for i in dict_fpr.values()],
        "FNR": [i for i in dict_fnr.values()]
    })
    print(data)
    # data.to_csv(os.path.join(CWD, "save_result", "%s_acc.csv" % os.path.basename(DIR)))
    if VERSION is not None:
        data.to_excel(os.path.join(CWD, "save_result", f"v.{VERSION}", "%s_%s_acc.xlsx" % (os.path.basename(DIR), os.path.basename(ovn_model))))
    else:
        data.to_excel(os.path.join(CWD, "save_result", "%s_%s_acc.xlsx" % (os.path.basename(DIR), os.path.basename(ovn_model))))

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

    if VERSION is not None:
        save_file = os.path.join(CWD, "save_result", f"v.{VERSION}", "%s-%s.xlsx" % (os.path.basename(DIR), os.path.basename(ovn_model)))
    else:
        save_file = os.path.join(CWD, "save_result", "%s-%s.xlsx" % (os.path.basename(DIR), os.path.basename(ovn_model)))

    with pd.ExcelWriter(save_file) as writer:
        for k, data in dict_data.items():
            data.to_excel(writer, sheet_name=k)
    print("save done at: ", save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--eval_ds", help="Path to evaludate dataset",
                        type=str, default="/mnt/datadrive/quangdn/far/face_classification/data/eval_all_face_28-3108_2_align_filter_align_0.15")
    parser.add_argument("-w", "--weight", help="Path to openvino weight",
                        type=str, default="/mnt/data/quangdn/far/trained_models/v.0.2/112_Classify_Adam_Epoch_196_Batch_20580_97.703_99.532_Time_1659991050.0512657_checkpoint.xml")
    parser.add_argument("-v", "--version", help="Version of weight if version is choosed, weight will auto loaded CHOICES: ['-1', '0.0', '0.1', '0.2', '0.3', '0.4', '1.0', '1.1', '1.2', '1.3'], DEFAULT: NONE",
                        choices=["-1", "0.0", "0.1", "0.2", "0.3", "0.4", "1.0", "1.1", "1.2", "1.3", "ncnn"], default="ncnn")
    parser.add_argument("--is_round", help="Round accuracy DEFAULT: TRUE", action="store_false")
    args = parser.parse_args()

    face_types_dict = dict()
    face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
    face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
    face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
    face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

    DIR = args.eval_ds
    IS_ROUND = args.is_round
    VERSION = args.version
    if VERSION is None:
        ovn_model = args.weight
    else:
        save_result_dir = os.path.join(CWD, "save_result", f"v.{VERSION}")
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)

        if VERSION == "ncnn":
            bin_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.bin"
            param_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.param"
            ovn_model = bin_path
            feat_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.bin_1.pkl"

    predict_cases = np.zeros([3, 3])

    times = []

    INPUT_SIZE = (112, 112)

    net = init_ncnn(bin_path, param_path)

    num_threads = 6
    target_size = 112
    print("Loading model Done...")

    dict_all_result = {}
    list_all_label = []
    list_all_pred = []
    dict_pred_output = {}
    for subdir, dirs, files in os.walk(DIR):
        dict_subdir_result = {}
        try:
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
                output = ncnn_infer(face, net)
                times.append(time.time() - t)
                # print("==> out: ", [out0, out1])
                dict_pred_output[file_path] = output
                multitask_to_true_false_cases(file_path, output, dict_subdir_result, list_all_pred)
                # single_task_to_false_cases(filename, output[0][0])
        except KeyboardInterrupt:
            print("Keyboard Interrupting...")
            # with open(feat_path, "wb") as f:
            #     pickle.dump(dict_pred_output, f)
            break
        dict_all_result[os.path.basename(subdir)] = dict_subdir_result

    with open(feat_path, "wb") as f:
        pickle.dump(dict_pred_output, f)
    print("save feature at: ", feat_path)

    cal_accuracy_params_for_cm(predict_cases, is_round=IS_ROUND)
    # print(list_all_label, list_all_pred)
    try:
        cal_precision_recall(list_all_label, list_all_pred)
    except Exception as e:
        print("error e: ", e)

    print("AVG time:", np.array(times).mean())
    df_cm = pd.DataFrame(predict_cases, columns=["Glass\npredicted", "Mask\npredicted", "Normal\npredicted"],
                        index=["Glass", "Mask", "Normal"])
    fig = plt.figure(figsize=(3, 3))
    sn.heatmap(df_cm, annot=True, fmt=".5g")
    fig.tight_layout()
    # plt.show()
    if VERSION is not None:
        plt.savefig(os.path.join(CWD, "save_result", f"v.{VERSION}", "%s_%s.png" % (os.path.basename(DIR), os.path.basename(ovn_model))))
    else:
        plt.savefig(os.path.join(CWD, "save_result", "%s_%s.png" % (os.path.basename(DIR), os.path.basename(ovn_model))))

    # subdir = [os.path.basename(sd) for sd in glob.glob("%s/*" % DIR) if os.path.isdir(sd)]
    # print(subdir)

    # print(dict_all_result.keys(), [len(v) for k, v in dict_all_result.items()])
    save_result(dict_all_result)
