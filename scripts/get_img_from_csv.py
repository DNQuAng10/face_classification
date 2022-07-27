"""
after get random index, get image
"""
import os
import pandas as pd
import shutil
import tqdm


def get_img_from_csv(type="glasses"):
    csv_path = "/home/dnq/Working/14-far/face_classification/scripts/get_rd_%s.csv" % type
    ds_path = "/mnt/d/Working/data/14-far/face-classification/aligned/%s" % type
    new_ds_path = "/mnt/d/Working/data/14-far/face-classification/aligned_get_rd/%s" % type

    if not os.path.exists(new_ds_path):
        os.makedirs(new_ds_path)

    data = pd.read_csv(csv_path, header=None)
    # print(data[1])
    c = 0
    for img_name in tqdm.tqdm(data[1]):
        old_path = os.path.join(ds_path, img_name)
        new_path = os.path.join(new_ds_path, img_name)
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
            # print(f"--> copy '{old_path}' to '{new_path}'")
            c += 1
    print("number of image name: ", len(data))
    print("copied: ", c)

if __name__ == "__main__":
    get_img_from_csv(type="glasses")
    get_img_from_csv(type="mask")
    get_img_from_csv(type="normal")
