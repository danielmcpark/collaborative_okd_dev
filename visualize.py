import torch
import numpy as np
import matplotlib.pyplot as plt

load_value = torch.load("./csv/pcsc/aa")
load_value = load_value.detach().cpu().numpy()
print(load_value.shape)
print(load_value.transpose().shape)
