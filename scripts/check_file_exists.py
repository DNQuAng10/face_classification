import glob
import os
import pandas as pd
import tqdm
import sys

csv_path = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/normal_FP_from_mask_FP.csv"
# csv_path = "/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/scripts/glasses_FP_from_mask_FP.csv"
data = pd.read_csv(csv_path)

ck_dir = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd/mask_FP"
ck_dir = "/home/dnq/data/FAR/glass_mask_classify/aligned/aligned/mask"

list_img = [os.path.basename(i) for i in glob.glob("%s/*" % ck_dir)]
print(list_img[0])
c = 0
for img_path in tqdm.tqdm(data["list_img"]):
    img_name = os.path.basename(img_path)
    # ck = [1 if img_name not in i else 0 for i in list_img]
    # ck_sum = sum(ck)
    # if ck_sum != 0:
    #     # print(img_name)
    #     c += 1
    if img_name not in list_img:
        c += 1
print(c)