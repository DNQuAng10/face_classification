"""
Statistics number of image 
"""
import argparse
import glob
import os
import pandas as pd

def get_csv_file_from_dir(dir_path, csv_name):
    list_img = glob.glob("%s/*" % dir_path)
    print(len(list_img))

    data = pd.DataFrame(data={
        "list_img": list_img
    })
    data.to_csv("/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/scripts/%s.csv" % csv_name)

if __name__ == "__main__":
    dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/eval_ds/glasses_FP_from_mask_FP_new"
    csv_name = "glasses_FP_from_mask_FP_new"
    get_csv_file_from_dir(dir_path, csv_name)

    # dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/glasses_FP_from_normal_FP"
    # csv_name = "glasses_FP_from_normal_FP"
    # get_csv_file_from_dir(dir_path, csv_name)
    
    # dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/normal_FP_from_glass_FP"
    # csv_name = "normal_FP_from_glass_FP"
    # get_csv_file_from_dir(dir_path, csv_name)

    # dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/normal_FP_from_mask_FP"
    # csv_name = "normal_FP_from_mask_FP"
    # get_csv_file_from_dir(dir_path, csv_name)

    # dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/mask_FP_from_normal_FP"
    # csv_name = "mask_FP_from_normal_FP"
    # get_csv_file_from_dir(dir_path, csv_name)

    # dir_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/eval_ds/mask_FP_from_glass_FP"
    # csv_name = "mask_FP_from_glass_FP"
    # get_csv_file_from_dir(dir_path, csv_name)