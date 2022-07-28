import glob
import os 
import pandas as pd
import pathlib
import shutil
import tqdm
import random

subeval = "/home/dnq/data/FAR/glass_mask_classify/eval_ds_fr_aligned/png/aligned_get_rd_small/summary"
list_img = glob.glob("%s/*" %subeval)

NUMBER = 6000

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
list_img_choose = [list_img[i] for i in choose]

data = pd.DataFrame(data={
    "list_img": list_img_choose
})
data.to_csv("/home/dnq/Working/FWorking/14-far/glass_mask_normal_face_classify/test/face_classification/png/sub_evaluate.csv")

dst_subeval = subeval.replace("summary", "sub_eval")
if not os.path.exists(dst_subeval):
    os.makedirs(dst_subeval)

c = 0
for path in tqdm.tqdm(list_img_choose):
    img_name = os.path.basename(path)
    dst_path = os.path.join(dst_subeval, img_name)
    if not os.path.exists(dst_path):
        shutil.copy(path, dst_path)
        c += 1

print("Number: ", len(list_img_choose))
print("copied: ", c)