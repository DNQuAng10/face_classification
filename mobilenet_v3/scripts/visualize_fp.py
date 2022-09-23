"""
visualize false positive 
"""
import argparse
import cv2 as cv
import os
import pandas as pd
import pathlib
import tqdm
import random

CWD = pathlib.Path(__file__).resolve().parents[1]


def save_fn_img(save_file, type="glasses"):
    data = pd.read_excel(save_file, sheet_name=type, engine="openpyxl")
    # print(data)
    file_name = os.path.basename(save_file).replace(pathlib.Path(save_file).suffix, "")
    try:
        eval_ds_name = file_name.split("-")[0]
        weight_name = file_name.split("-")[1]
    except:
        eval_ds_name = file_name
        weight_name = "best"        

    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        if data["label"][i] != data["pred"][i]:
            if device == "gpu3":
                img = cv.imread(data["image path"][i])
            elif device == "gpu2":
                img = cv.imread(str(data["image path"][i]).replace("datadrive", "data"))
            # text = f"label: {data["label"][i]} pred: {data["pred"][i]}"
            # cv.putText(img, text, ())
            save_dir = os.path.join(
                CWD, "visualize", 
                eval_ds_name, weight_name, 
                "{}_FN".format(type), "to_{}".format(data["pred"][i]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            save_img_path = os.path.join(save_dir, "{}.jpg".format(os.path.basename(data["image path"][i])))
            cv.imwrite(save_img_path, img)
            # print("save at: ", save_img_path)
    print("Save visualize at: ", os.path.basename(os.path.basename(save_dir)))

def save_tp_img(save_file, type="glasses"):
    data = pd.read_excel(save_file, sheet_name=type, engine="openpyxl")
    # print(data)
    file_name = os.path.basename(save_file).replace(pathlib.Path(save_file).suffix, "")
    try:
        eval_ds_name = file_name.split("-")[0]
        weight_name = file_name.split("-")[1]
    except:
        eval_ds_name = file_name
        weight_name = "best"        

    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        if data["label"][i] == data["pred"][i]:
            if device == "gpu3":
                img = cv.imread(data["image path"][i])
            elif device == "gpu2":
                img = cv.imread(str(data["image path"][i]).replace("datadrive", "data"))
            # text = f"label: {data["label"][i]} pred: {data["pred"][i]}"
            # cv.putText(img, text, ())
            save_dir = os.path.join(
                CWD, "visualize", 
                eval_ds_name, weight_name, 
                "{}_TP".format(type))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            save_img_path = os.path.join(save_dir, "{}.jpg".format(os.path.basename(data["image path"][i])))
            cv.imwrite(save_img_path, img)
            # print("save at: ", save_img_path)
    print("Save visualize at: ", os.path.basename(os.path.basename(save_dir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to save classification result excel files", type=str)
    parser.add_argument("-e", "--gpu", help="Run on ['gpu2', 'gpu3'] DEFAULT: gpu2", choices=["gpu2", "gpu3"], type=str, default="gpu2")
    parser.add_argument("-m", "--mode", help="Visualize TP or FP ['tp', 'fp']", choices=["tp", "fp"], default="fp")
    args = parser.parse_args()
    # save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/eval.xlsx"
    # save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/sub_eval_labeled.xlsx"
    # save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/sub_eval_labeled-112_Classify_Adam_Epoch_197_Batch_8077_95.258_98.877_Time_1659628469.711623_checkpoint.xml.xlsx"
    # save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/sub_eval_labeled-112_Classify_Adam_Epoch_196_Batch_20580_97.703_99.532_Time_1659991050.0512657_checkpoint.xml.xlsx"
    # save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/sub_eval_labeled_1-112_Classify_Adam_Epoch_188_Batch_21620_98.263_99.347_Time_1660073353.1392345_checkpoint.xml.xlsx"
    # # save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/bast/v.datnt/sub_eval_labeled-112_Classify_Adam_Epoch_75_Batch_6750_95.657_97.667_Time_1634623345.5846994_checkpoint.xml.xlsx"
    # save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/bast/v.0.4/sub_eval_labeled_1-112_Classify_Adam_Epoch_188_Batch_21620_98.263_99.347_Time_1660073353.1392345_checkpoint.xml.xlsx"

    save_file = args.input
    device = args.gpu
    mode = args.mode

    for i in ["glasses", "mask", "normal"]:
        if mode == "fp":
            save_fn_img(save_file, type=i)
        elif mode == "tp":
            save_tp_img(save_file, type=i)