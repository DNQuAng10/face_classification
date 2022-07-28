"""
get random index a number of TP image from TP each class
"""

import glob
import os
import pandas as pd
import random

GLASSES_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/glasses_TP"
MASK_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/mask_TP"
NORMAL_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/normal_TP"
NUMBER = 1500

def get_rd_tp(type="glasses"):
    if type == "glasses":
        dir_path = GLASSES_TP
    elif type == "mask":
        dir_path = MASK_TP
    elif type == "normal":
        dir_path = NORMAL_TP
    
    print("directory: ", dir_path)
    list_img = glob.glob("%s/*" % dir_path)

    choose = random.choices([i for i in range(len(list_img) - 1)], [1 for i in range(len(list_img) - 1)], k=NUMBER)
    choose = list(set(choose))
    if len(choose) != NUMBER:
        while len(choose) < NUMBER:
            x = random.randint(0, len(list_img) - 1)
            if x not in choose:
                choose.append(x)
            if len(choose) == NUMBER:
                break
    
    choose = list(set(choose))
    print("Number of choose: ", len(choose))

    list_tp_img = [os.path.basename(list_img[i]) for i in choose]
    print("Number: ", len(list_tp_img))

    data = pd.DataFrame(data={
        "list_img": list_tp_img
    })
    data.to_csv("/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/scripts/get_rd_tp_%s.csv" % type)
    print("save done...")

if __name__ == "__main__":
    get_rd_tp(type="glasses")
    # NUMBER = 2000
    # get_rd_tp(type="mask")
    # get_rd_tp(type="normal")