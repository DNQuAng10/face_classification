import numpy as np

x = np.array([[10, 1, 2], [0, 11, 12], [2, 3, 12]])
print(x)

def cal_accuracy_params(cm):
    shape, shape = cm.shape
    dict_tp = {}
    dict_fp = {}
    dict_tn = {}
    dict_fn = {}
    for i in range(shape):
        for j in range(shape):
            if i == j:
                dict_tp[i] = cm[i][j]
                dict_fp[i] = sum([cm[r][j] for r in range(shape) if r != i])
                dict_fn[i] = sum([cm[i][c] for c in range(shape) if c != i])
                dict_tn[i] = sum([cm[rc][rc] for rc in range(shape) if rc != i])
    
    dict_precision = {}
    dict_recall = {}
    dict_fpr = {}
    for i in range(shape):
        dict_precision[i] = dict_tp[i] / (dict_tp[i] + dict_fp[i])
        dict_recall[i] = dict_tp[i] / (dict_tp[i] + dict_fn[i])
        dict_fpr[i] = dict_fp[i] / (dict_fp[i] + dict_tn[i])
    
    print("TP: ", dict_tp)
    print("FP: ", dict_fp)
    print("TN: ", dict_tn)
    print("FN: ", dict_fn)
    print("Precision: ", dict_precision)
    print("Recall: ", dict_recall)
    print("FPR: ", dict_fpr)

if __name__ == "__main__":
    cal_accuracy_params(x)
