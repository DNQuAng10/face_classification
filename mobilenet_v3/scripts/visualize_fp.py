"""
visualize false positive 
"""
import cv2 as cv
import os
import pandas as pd
import pathlib
import tqdm

CWD = pathlib.Path(__file__).resolve().parents[1]


def save_fp_img(save_file, type="glasses"):
    data = pd.read_excel(save_file, sheet_name=type)
    # print(data)

    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        if data["label"][i] != data["pred"][i]:
            img = cv.imread(data["image path"][i])
            # text = f"label: {data["label"][i]} pred: {data["pred"][i]}"
            # cv.putText(img, text, ())
            save_dir = os.path.join(
                CWD, "visualize", 
                os.path.basename(save_file).replace(pathlib.Path(save_file).suffix, ""), 
                "{}_FP".format(data["pred"][i]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            save_img_path = os.path.join(save_dir, "{}.jpg".format(os.path.basename(data["image path"][i])))
            cv.imwrite(save_img_path, img)
            # print("save at: ", save_img_path)

if __name__ == "__main__":
    save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/eval.xlsx"
    save_file = "/home/quangdn/far/face_classification/Torch_MobilenetV3/save_result/sub_eval_labeled.xlsx"

    for i in ["glasses", "mask", "normal"]:
        save_fp_img(save_file, type=i)
    