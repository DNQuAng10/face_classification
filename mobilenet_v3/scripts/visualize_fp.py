"""
visualize false positive 
"""
import cv2 as cv
import os
import pandas as pd
import pathlib
import tqdm
import random

CWD = pathlib.Path(__file__).resolve().parents[1]


def save_fp_img(save_file, type="glasses"):
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
            img = cv.imread(data["image path"][i])
            # text = f"label: {data["label"][i]} pred: {data["pred"][i]}"
            # cv.putText(img, text, ())
            save_dir = os.path.join(
                CWD, "visualize", 
                eval_ds_name, weight_name, 
                "{}_FP".format(data["pred"][i]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            save_img_path = os.path.join(save_dir, "{}.jpg".format(os.path.basename(data["image path"][i])))
            cv.imwrite(save_img_path, img)
            # print("save at: ", save_img_path)

if __name__ == "__main__":
    save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/eval.xlsx"
    save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/sub_eval_labeled.xlsx"
    save_file = "/home/quangdn/far/face_classification/mobilenet_v3/save_result/sub_eval_labeled-112_Classify_Adam_Epoch_197_Batch_8077_95.258_98.877_Time_1659628469.711623_checkpoint.xml.xlsx"

    for i in ["glasses", "mask", "normal"]:
        save_fp_img(save_file, type=i)
    