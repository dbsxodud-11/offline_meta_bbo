import os
from importlib.machinery import SourceFileLoader

import torch

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()

def stack_tensor(x, num_samples, dim):
    return torch.stack([x] * num_samples, dim)
