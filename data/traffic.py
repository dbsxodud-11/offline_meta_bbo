import pickle

import torch
import torch.multiprocessing as mp
import pandas as pd
from attrdict import AttrDict


class DataSampler():
    def __init__(self, data_path, dtype, device):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        data.x = data.x.to(dtype=dtype, device=device)
        data.y = data.y.to(dtype=dtype, device=device)
        self.data = data

    def sample(self, batch_size, max_num_points):
        batch = AttrDict()
        num_context = torch.randint(low=10, high=max_num_points-10, size=[1]).item()
        num_target = torch.randint(low=10, high=max_num_points-num_context, size=[1]).item()

        num_points = num_context + num_target
        batch.idx = torch.randint(0, self.data.x.shape[-2], size=(batch_size, num_points, 1), device=self.data.x.device)

        batch.x = self.data.x.gather(1, batch.idx.repeat(1, 1, self.data.x.shape[-1]))
        batch.y = self.data.y.gather(1, batch.idx.repeat(1, 1, self.data.y.shape[-1]))

        batch.x_context = batch.x[..., :num_context, :]
        batch.y_context = batch.y[..., :num_context, :]

        batch.x_target = batch.x[..., num_context:, :]
        batch.y_target = batch.y[..., num_context:, :]
        return batch
