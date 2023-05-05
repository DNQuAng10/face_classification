import argparse
import time

import os
import cv2
import glob
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import torch
import tqdm

CWD = pathlib.Path(__file__).resolve().parent
# from modules.LandmarksCropper import LandmarksCroper
# from modules.TDDFA.TDDFA import TDDFA_Blob
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from MobilenetV3 import mobilenetv3_small_multitask


CLASSES = ["glasses", "mask", "normal"]

def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    state_dict = state_dict["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    print("state_dict all_keys: ", all_keys)
    # for k in all_keys:
    #     if k.startswith('module.'):
    #         state_dict[k[7:]] = state_dict.pop(k)
    # print("state_dict", state_dict.keys())
    model_dict = model.state_dict()
    model_dict_keys = ["module." + i for i in model_dict.keys()]
    print("model_dict: ", len(model_dict), type(model_dict), "state_dict: ", len(state_dict))
    print([i for i in model_dict.keys()][: 10])
    print([i for i in state_dict.keys()][: 10])
    # print("model dict", model_dict.keys())
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    # print("pretrained dict", pretrained_dict.keys())
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in model_dict.keys() if k not in pretrained_dict.keys()}
        print("not loaded keys:", len(not_loaded_keys))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



def multitask_to_true_false_cases(file_path, output, dict_results: dict=None, list_pred: list=None):
    # glasses, mask, normal = 0, 1, 2
    global face_types_dict, predict_cases
    output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
    
    dict_result = {}
    dir_name = os.path.basename(os.path.dirname(file_path))
    print("output_classify: ", output_classify)
    print("dir:", dir_name)
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
                        choices=["-1", "0.0", "0.1", "0.2", "0.3", "0.4", "1.0", "1.1", "1.2", "1.3", "2.5"], default="2.5")
    parser.add_argument("--is_round", help="Round accuracy DEFAULT: TRUE", action="store_false")
    args = parser.parse_args()

    face_types_dict = dict()
    face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
    face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
    face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
    face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

    # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval"
    # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_1"
    # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_png_0"
    # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/eval_png_1"
    # # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/test_eval_png_1"
    # # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/new_aligned"
    # DIR = "/mnt/datadrive/quangdn/far/face_classification/data/sub_eval_labeled"
    # DIR = "/mnt/data/quangdn/far/face_classification/data/sub_eval_labeled_1"
    DIR = args.eval_ds
    IS_ROUND = args.is_round
    VERSION = args.version
    if VERSION is None:
        ovn_model = args.weight
    else:
        save_result_dir = os.path.join(CWD, "save_result", f"v.{VERSION}")
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
            
        if VERSION == "-1":
            # ovn_model = "/home/quangdn/far/face_classification/models/112_Classify_Adam_Epoch_75_Batch_6750_95.657_97.667_Time_1634623345.5846994_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.datnt/112_Classify_Adam_Epoch_75_Batch_6750_95.657_97.667_Time_1634623345.5846994_checkpoint.xml"
        elif VERSION == "0.0":
            # v.0.0
            ovn_model = "/mnt/datadrive/quangdn/far/trained_models/112_Classify_Adam_Epoch_197_Batch_8077_95.258_98.877_Time_1659628469.711623_checkpoint.xml"
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/112_Classify_Adam_Epoch_200_Batch_8200_95.426_99.045_Time_1659628613.133665_checkpoint.xml"
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/112_Classify_Adam_Epoch_180_Batch_7380_95.291_98.996_Time_1659626529.3278854_checkpoint.xml"
        elif VERSION == "0.1":
            # v.0.1
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/v.0.1/112_Classify_Adam_Epoch_50_Batch_2600_94.862_96.474_Time_1659690931.2707672_checkpoint.xml"
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/v.0.1/112_Classify_Adam_Epoch_183_Batch_9516_96.723_98.687_Time_1659719729.6839921_checkpoint.xml"
            ovn_model = "/mnt/datadrive/quangdn/far/trained_models/v.0.1/112_Classify_Adam_Epoch_199_Batch_10348_96.533_98.920_Time_1659721905.793932_checkpoint.xml"
        elif VERSION == "0.2":
            # v.0.2
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.2/112_Classify_Adam_Epoch_196_Batch_20580_97.703_99.532_Time_1659991050.0512657_checkpoint.xml"
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/v.0.2/112_Classify_Adam_Epoch_188_Batch_19740_97.374_99.501_Time_1659988508.269691_checkpoint.xml"
            # ovn_model = "/mnt/datadrive/quangdn/far/trained_models/v.0.2/112_Classify_Adam_Epoch_97_Batch_10185_96.474_99.167_Time_1659965059.083403_checkpoint.xml"
        elif VERSION == "0.3":
            # # v.0.3
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.3/112_Classify_Adam_Epoch_191_Batch_24066_98.448_99.388_Time_1660044401.2665846_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.3/112_Classify_Adam_Epoch_197_v.0.3_Batch_24822_98.504_99.404_Time_1660045235.0492127_checkpoint.xml"
        elif VERSION == "0.4":
            # # v.0.4
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.4/112_Classify_Adam_Epoch_188_Batch_21620_98.263_99.347_Time_1660073353.1392345_checkpoint.xml"
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.4/112_Classify_Adam_Epoch_189_Batch_21735_96.901_99.378_Time_1660073492.408608_checkpoint.xml"
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.0.4/112_Classify_Adam_Epoch_191_Batch_21965_98.103_99.404_Time_1660073736.3311913_checkpoint.xml"
        elif VERSION == "1.0":
            # v.1.0
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.0/112_Classify_Adam_Epoch_198_Batch_22176_97.728_99.522_Time_1660156442.7692828_checkpoint.xml"
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.0/112_Classify_Adam_Epoch_185_Batch_20720_97.985_99.465_Time_1660154832.307503_checkpoint.xml"
        elif VERSION == "1.1":
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.1/112_Classify_Adam_Epoch_147_Batch_16464_73.067_64.325_Time_1660644692.3265386_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.1/112_Classify_Adam_Epoch_197_Batch_22064_71.371_64.253_Time_1660651715.6633825_checkpoint.xml"
        elif VERSION == "1.2":
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.2/112_Classify_Adam_Epoch_191_Batch_21774_97.959_99.486_Time_1660750499.4749315_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.2/112_Classify_Adam_Epoch_195_Batch_22230_98.031_99.409_Time_1660751043.7966943_checkpoint.xml"
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.2/112_Classify_Adam_Epoch_198_Batch_22572_97.034_99.486_Time_1660751451.8433118_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.2/112_Classify_Adam_Epoch_200_Batch_22800_97.579_99.414_Time_1660751719.103265_checkpoint.xml"
        elif VERSION == "1.3":
            # ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.3/112_Classify_Adam_Epoch_165_Batch_19140_98.407_99.316_Time_1660819704.0766041_checkpoint.xml"
            ovn_model = "/mnt/data/quangdn/far/trained_models/v.1.3/112_Classify_Adam_Epoch_200_Batch_23200_97.903_99.378_Time_1660824643.8839684_checkpoint.xml"
        elif VERSION == "2.5":
            ovn_model = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.pth"
            feat_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.pth.pkl"

    predict_cases = np.zeros([3, 3])

    times = []

    INPUT_SIZE = (112, 112)    

    print("MODEL: ", ovn_model)
    model = mobilenetv3_small_multitask()

    print("backbone path", ovn_model)
    load_state_dict(model, torch.load(ovn_model, map_location="cpu"))
    # backbone.cuda()
    model.eval()
    model.cuda(device=0)
    print("Loading model Done...")

    dict_all_result = {}
    list_all_label = []
    list_all_pred = []
    dict_pred_output = {}
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
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img_t = np.transpose(face, (2, 0, 1))
            img_t = img_t - 127.5
            img_t /= 127.5
            img_expand_dim = np.expand_dims(img_t, axis=0).astype(np.float32)
            test_torch = torch.from_numpy(img_expand_dim).cuda(device=0)
            
            # face = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)
            w, h, c = face.shape
            if face is None:
                continue
            t = time.time()
            with torch.no_grad():
                res00, res01 = model(test_torch)
            output = np.array([res00.cpu().numpy(), res01.cpu().numpy()])
            print("output: ", output, output.shape)
            dict_pred_output[file_path] = output
            times.append(time.time() - t)
            multitask_to_true_false_cases(file_path, output, dict_subdir_result, list_all_pred)
            # single_task_to_false_cases(filename, output[0][0])
        dict_all_result[os.path.basename(subdir)] = dict_subdir_result

    import pickle
    with open(feat_path, "wb") as f:
        pickle.dump(dict_pred_output, f)
    print("save feature at: ", feat_path)

    cal_accuracy_params_for_cm(predict_cases, is_round=IS_ROUND)

    cal_precision_recall(list_all_label, list_all_pred)

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
