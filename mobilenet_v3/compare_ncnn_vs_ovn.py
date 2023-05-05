import numpy as np
import pickle
import tqdm


if __name__ == "__main__":
    ncnn_feat_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.bin_1.pkl"
    ovn_feat_path = "/home/quangdn/frs_mobile/gitlab/face_classification/mobilenet_v3/models/v.2.5/ovn/112_Classify_Adam_Epoch_197_Batch_21079_78.955_86.377_Time_1664504358.4962163_checkpoint.xml.pkl"

    with open(ncnn_feat_path, "rb") as f:
        ncnn_feat = pickle.load(f)
    
    with open(ovn_feat_path, "rb") as f:
        ovn_feat = pickle.load(f)

    print(len(ncnn_feat), len(ovn_feat))    

    c = 0
    for k in tqdm.tqdm(ncnn_feat.keys()):
        # if "normal" not in k:
        #     continue
        ncnn_out = ncnn_feat[k]
        ovn_out = ovn_feat[k]
        diff = np.max(np.abs(ncnn_out - ovn_out))
        if diff > 0.0001:
            print("key: ", k)
            print("ncnn out: ", ncnn_out)
            print("ovn out: ", ovn_out)
            print("diff: ", diff)
            c += 1
    print(c)