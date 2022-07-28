"""
get image again beacause change jpg dataset to png dataset.
"""
import os 
import pandas as pd
import pathlib
import shutil
import tqdm


GLASSES_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_png/glasses"
MASK_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_png/mask"
NORMAL_TP = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_png/normal"

def get_tp_img(type="glasses"):
    if type == "glasses":
        dir_path = GLASSES_TP
    elif type == "mask":
        dir_path = MASK_TP
    elif type == "normal":
        dir_path = NORMAL_TP
    dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/eval_ds/%s_TP" % type
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    if type == "glasses" or type == "normal":
        csv_path = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/get_rd_tp_%s.csv" % type
    else:
        csv_path = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/get_rd_tp_mask_2000.csv"

    data = pd.read_csv(csv_path)
    list_img_name = data["list_img"]

    c = 0
    for img_name in tqdm.tqdm(list_img_name, total=len(list_img_name)):
        img_name = img_name.replace(pathlib.Path(img_name).suffix, ".png")
        src = os.path.join(dir_path, img_name)
        dst = os.path.join(dst_path, img_name)

        if not os.path.exists(dst):
            shutil.copy(src, dst)
            c += 1
    
    print("Number: ", len(list_img_name))
    print("Copied: ", c)

def get_fp_img(type="glasses"):
    if type == "glasses":
        # glasses_fp_from_mask
        fp_from_0 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/glasses_FP_from_mask_FP_new.csv"
        # glasses_fp_from_normal
        fp_from_1 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/glasses_FP_from_normal_FP.csv"
        dir_path_0 = MASK_TP
        dir_path_1 = NORMAL_TP
    elif type == "mask":
        # mask_fp_from_glasses
        fp_from_0 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/mask_FP_from_glass_FP.csv"
        # mask_fp_from_normal
        fp_from_1 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/mask_FP_from_normal_FP.csv"
        dir_path_0 = GLASSES_TP
        dir_path_1 = NORMAL_TP
    elif type == "normal":
        # normal_fp_from_glasses
        fp_from_0 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/normal_FP_from_glass_FP.csv"
        # normal_fp_from_mask
        fp_from_1 = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/normal_FP_from_mask_FP.csv"
        dir_path_0 = GLASSES_TP
        dir_path_1 = MASK_TP
    dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/eval_ds/"

    def copy_img(data, fp_from, dir_path, dst_path):
        print("--> directory:  ", dir_path)
        c = 0
        for img_path in data["list_img"]:
            img_name = os.path.basename(img_path)
            img_name = img_name.replace(pathlib.Path(img_name).suffix, ".png")
            
            dst_dir = str(os.path.basename(fp_from))
            dst_dir = dst_dir.replace(pathlib.Path(dst_dir).suffix, "")
            dst_path_1 = os.path.join(dst_path, dst_dir)
            if not os.path.exists(dst_path_1):
                os.makedirs(dst_path_1)

            src_path = os.path.join(dir_path, img_name)
            dst_ = os.path.join(dst_path_1, img_name)

            if not os.path.exists(dst_):
                shutil.copy(src_path, dst_)
                c += 1
        print("Number: ", len(data["list_img"]))
        print("Copied: ", c)

    data_0 = pd.read_csv(fp_from_0)
    data_1 = pd.read_csv(fp_from_1)

    copy_img(data_0, fp_from_0, dir_path_0, dst_path)
    copy_img(data_1, fp_from_1, dir_path_1, dst_path)


if __name__ == "__main__":
    # get_tp_img("glasses")
    # get_tp_img("mask")
    # get_tp_img("normal")

    get_fp_img("glasses")
    # get_fp_img("mask")
    # get_fp_img("normal")
