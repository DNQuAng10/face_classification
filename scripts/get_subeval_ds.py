"""
get number of image from aligned dataset that is filtered by get 1 image on each session (mask only take 4000 images)
"""
import glob
import os
import pandas as pd
import pathlib
import tqdm
import shutil

def get_img(type="glasses"):
    if type == "glasses":
        src_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd_small/glasses"
        dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_get_rd_small/glasses"
    elif type == "mask":
        src_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd_small/mask_small"
        dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_get_rd_small/mask_small"
    elif type == "normal":
        src_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd_small/normal"
        dst_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_get_rd_small/normal"

    get_path = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_png/%s" % type 
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    list_img = [os.path.basename(i) for i in glob.glob("%s/*" % src_path)]

    c = 0
    list_png_img = []
    for img_name in tqdm.tqdm(list_img):
        try:
            img_name = img_name.replace(pathlib.Path(img_name).suffix, ".png")
            src = os.path.join(get_path, img_name)
            dst = os.path.join(dst_path, img_name)
            list_png_img.append(img_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                c += 1
        except Exception as e:
            print(e)
    print("Number: ", len(list_img))
    print("copied: ", c)

    data = pd.DataFrame(data={
        "list_img": list_png_img
    })
    data.to_csv("/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/png/get_rd_small_%s.csv" % type)


if __name__ == "__main__":
    get_img("glasses")
    get_img("mask")
    get_img("normal")