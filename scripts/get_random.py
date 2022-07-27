"""
get random index a image on each session in dataset
"""

import glob
import os
import pandas as pd
import tqdm
import random


def get_random(dir):
    print("---> Directory: ", os.path.basename(dir))
    list_img = glob.glob("%s/*" % dir)
    print(len(list_img))

    list_img_name = [os.path.basename(i) for i in list_img]
    list_img_name_1 = ["_".join(i.split("_")[:2]) for i in list_img_name]
    # print(list_img_name_1[0])
    print("number name: ", len(list_img_name_1))
    list_img_name_loop = list(set(list_img_name_1))
    print("number of loop name: ", len(list_img_name_loop))

    dict_loop = {}
    for loop in tqdm.tqdm(list_img_name_loop, total=len(list_img_name_loop)):
        list_loop = []
        for i in list_img_name:
            if loop in i:
                list_loop.append(i)
        dict_loop[loop] = list_loop

    # print(len(dict_loop.keys()))
    # print(len(list(dict_loop.values())[0]))

    data = pd.DataFrame(data={
        "img_name": dict_loop.keys(),
        "number": [len(i) for i in dict_loop.values()]
    })

    # print(data)
    data.to_csv("/home/dnq/Working/14-far/face_classification/scripts/stat_%s.csv" % os.path.basename(dir))

    random.seed(123456)
    list_random = []
    for k, v in dict_loop.items():
        x = random.randint(0, len(v) - 1)
        # print(x)
        list_random.append(v[x])

    data = pd.DataFrame(data=list_random)
    data.to_csv("/home/dnq/Working/14-far/face_classification/scripts/get_rd_%s.csv" % os.path.basename(dir), header=None)

    print("DONE...")

if __name__ == "__main__":
    path = "/mnt/d/Working/data/14-far/face-classification/aligned"
    list_dir = glob.glob("%s/*" % path)

    for dir in list_dir:
        get_random(dir)
