import os
from os.path import join as pjoin
import pickle
import argparse
import random

import yaml
import torch
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict

from env.sumo_env import parallelize_sumo


if __name__ == "__main__":
    # Argument Passing
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="2by2")
    parser.add_argument("--scheme", type=str, default="comb")
    parser.add_argument("--num_worker", type=int, default=5)
    parser.add_argument("--root", type=str, default='.')
    args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    network = args.network
    scheme = args.scheme
    num_worker = args.num_worker
    root = args.root
    if not os.path.isdir(f"./data/{network}"):
        os.makedirs(f"./data/{network}")

    # Settings
    settings = yaml.load(open(pjoin(root, "config", network, "collect_data.yaml"), "r"),
                            Loader=yaml.SafeLoader)
    torch.manual_seed(settings["seed"])

    # 1. make & save raw data for training
    raw_data = AttrDict()
    raw_data.x = torch.zeros(size=(settings["num_scenarios"], settings["num_samples_per_scenario"], settings["raw_input_dim"]))
    raw_data.y = torch.zeros(size=(settings["num_scenarios"], settings["num_samples_per_scenario"], settings["raw_output_dim"]))
    
    for scenario in tqdm(range(settings["num_scenarios"])):
        x_task, y_task = parallelize_sumo(args, scenario, settings, num_worker, 'train')
        raw_data.x[scenario] = x_task
        raw_data.y[scenario] = y_task
    
    with open(f"./data/{network}/input_output_pair_{scheme}.pkl", "wb") as f:
        pickle.dump(raw_data, f)

    raw_data_val = AttrDict()
    raw_data_val.x = torch.zeros(size=(settings["num_val_scenario"], settings["num_samples_per_scenario"], settings["raw_input_dim"]))
    raw_data_val.y = torch.zeros(size=(settings["num_val_scenario"], settings["num_samples_per_scenario"], settings["raw_output_dim"]))

    for scenario in tqdm(range(settings["num_val_scenario"])):
        x_task, y_task = parallelize_sumo(args, scenario, settings, num_worker, 'valid')
        raw_data_val.x[scenario] = x_task
        raw_data_val.y[scenario] = y_task

    with open(f"./data/{network}/input_output_pair_valid_{scheme}.pkl", "wb") as f:
        pickle.dump(raw_data_val, f)

    data = AttrDict()
    data.x = raw_data.x
    data.y = torch.sum(raw_data.y, dim=-1, keepdim=True)

    data_val = AttrDict()
    data_val.x = raw_data_val.x
    data_val.y = torch.sum(raw_data_val.y, dim=-1, keepdim=True)

    data.y_min = data_val.y_min = min(data.y.min(), data_val.y.min())
    data.y_max = data_val.y_max = max(data.y.max(), data_val.y.max())

    data.y = (data.y - data.y_min) / (data.y_max - data.y_min)
    data_val.y = (data_val.y - data_val.y_min) / (data_val.y_max - data_val.y_min)
    
    with open(f"./data/{network}/traffic_data_{scheme}.pkl", "wb") as f:
        pickle.dump(data, f)

    with open(f"./data/{network}/traffic_data_valid_{scheme}.pkl", "wb") as f:
        pickle.dump(data_val, f)
