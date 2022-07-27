"""
after get random index, get TP image
"""
import os 
import pandas as pd
import shutil
import tqdm


GLASSES_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/glasses_TP"
MASK_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/mask_TP"
NORMAL_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/aligned_get_rd/normal_TP"

def get_img_from_tp_dir(type="glasses"):
    if type == "glasses":
        dir_path = GLASSES_TP
    elif type == "mask":
        dir_path = MASK_TP
    elif type == "normal":
        dir_path = NORMAL_TP
    dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/%s_TP" % type
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    csv_path = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/scripts/get_rd_tp_%s.csv" % type

    data = pd.read_csv(csv_path)
    list_img_name = data["list_img"]

    c = 0
    for img_name in tqdm.tqdm(list_img_name, total=len(list_img_name)):
        src = os.path.join(dir_path, img_name)
        dst = os.path.join(dst_path, img_name)

        if not os.path.exists(dst):
            shutil.copy(src, dst)
            c += 1
    
    print("Number: ", len(list_img_name))
    print("Copied: ", c)

if __name__ == "__main__":
    get_img_from_tp_dir("glasses")
    get_img_from_tp_dir("mask")
    get_img_from_tp_dir("normal")