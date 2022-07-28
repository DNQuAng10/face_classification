import glob
import os
import pandas as pd
import pathlib
import shutil
import random

mask_dir = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd_small/mask"
list_img = glob.glob("%s/*" % mask_dir)
print(len(list_img))

NUMBER = 1391

random.seed(123456)
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
print(choose[0:50])

list_choose_img = [os.path.basename(list_img[i]) for i in choose]

data = pd.DataFrame(data={
    "list_img": list_choose_img,
})
data.to_csv("/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/jpg/get_rd_mask_fp_1391.csv")

dst_dir = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/jpg/aligned_get_rd_small/mask_small"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

c = 0
for img_name in list_choose_img:
    src_path = os.path.join(mask_dir, img_name)
    dst_path = os.path.join(dst_dir, img_name)
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
        c += 1
print("Number: ", len(list_choose_img))
print("copied: ", c)