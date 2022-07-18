import torch
import torch.nn as nn
from torch.utils import data as Data
import d2l.torch as d2l
from stgcn import STGCN
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
args = parser.parse_args()

if __name__ == "__main__":
    batch_size = 16
    lr = 0.01
    num_classes = 60

    net = STGCN(num_classes, layout='ntu-rgb+d', in_channels=3, edge_importance_weighting=False, device=d2l.try_gpu())
    net.load_state_dict(torch.load("./first_test.torchmodel"))

    device = d2l.try_gpu()
    net.to(device)
    net.eval()

    raw_data = np.load(args.data_path)
    data = torch.tensor(raw_data.reshape(1, *raw_data.shape)).to(device)
    print(data.shape)
    print(net(data))
