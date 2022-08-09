import argparse
import os
import pathlib
import torch

from MobilenetV3 import mobilenetv3_small_multitask, mobilenetv3_small_singletask, mobilenetv3_large_multitask
from ECA_MobilenetV2 import eca_mobilenet_v2


def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pth_model", help="Path to PyTorch model", type=str, default=None)
    parser.add_argument("-o", "--onnx_model", help="Path to ONNX output model", type=str, default=None)
    args = parser.parse_args()

    PATH_TO_MODEL = args.pth_model
    assert PATH_TO_MODEL != None, "Path to PyTorch model cannot None"
    onnx_path = args.onnx_model
    if onnx_path is None:
        onnx_path = PATH_TO_MODEL.replace(pathlib.Path(PATH_TO_MODEL).suffix, ".onnx")

    model_name = os.path.basename(onnx_path)


    net = mobilenetv3_small_multitask()
    load_state_dict(net, torch.load(PATH_TO_MODEL, map_location="cpu")["model"])
    net.eval()
    dummy_input = torch.randn(1, 3, 112, 112)
    torch.onnx.export(net, dummy_input, onnx_path)
    # output_names=["0_glasses","1_mask","2_hat"])
    # output_names=["0_glasses","1_mask"])

    # Convert ONNX to OpenVino
    # os.system(
    #     "python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py --input_model {} --scale_values '[127.5,127.5,127.5]' --mean_values '[127.5,127.5,127.5]' --input_shape '[1,3,112,112]' --output_dir ~/Downloads/ --reverse_input_channels".format(onnx_path))

    # os.system("conda activate fas")
    os.system(
        f"mo --input_model {onnx_path} \
            --scale_values '[127.5,127.5,127.5]' \
            --mean_values '[127.5,127.5,127.5]' \
            --input_shape '[1,3,112,112]' \
            --output_dir {os.path.dirname(onnx_path)} \
            --reverse_input_channels")