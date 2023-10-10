import os
from os.path import join as pjoin
import pickle
import argparse
import pickle
import random

import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound

from utils.log import get_logger
from utils.misc import load_module
from env.sumo_env import SumoEnv


if __name__ == "__main__":
    # Argument Passing
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="2by2")
    parser.add_argument("--scheme", type=str, default="comb")
    parser.add_argument("--model", type=str, default='anp')
    parser.add_argument("--exp_id", type=str, default="trial1")
    parser.add_argument("--scenario_id", type=int, default=0)
    parser.add_argument("--root", type=str, default='.')
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    network = args.network
    scheme = args.scheme
    model = args.model
    exp_id = args.exp_id
    scenario_id = args.scenario_id
    seed = args.seed
    
    root = pjoin(args.root, "results", network, scheme, model, exp_id)
    if not os.path.isdir(root):
        os.makedirs(root)

    # Set Path for loading model, saving results, and logger
    model_load_path = pjoin(root, "ckpt.tar")
    results_save_path = pjoin(root, f"results_{args.scenario_id}.pkl")
    log_path = pjoin(root, f"test_{args.scenario_id}.log")
    logger = get_logger(log_path)

    # Set Hyperparameters
    settings = yaml.load(open(pjoin(args.root, "config", network, "test_settings.yaml"), "r"), Loader=yaml.SafeLoader)

    # Set network
    env = SumoEnv(network, scheme, scenario_id, args.root, run_type='test')
    dim, bounds, equality_constraints = env.get_constraints(dtype, device)

    # Load Data
    data_path = pjoin(args.root, "data", network, f"traffic_data_{scheme}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    y_min = data.y_min
    y_max = data.y_max  
    
    # Load Model
    model_cls = getattr(load_module(pjoin(args.root, "models", f"{args.model}.py")), args.model.upper())
    with open(pjoin(args.root, "results", network, scheme, model, exp_id, "model.yaml"), "r") as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).to(dtype=dtype, device=device)

    ckpt = torch.load(model_load_path)
    model.load_state_dict(ckpt.model)

    # Test
    test_results_overall = defaultdict(list)
    for num_test in range(settings["num_tests"]):
        test_results = {}
        
        # Set Seed for Reproduction
        seed = settings["seed"] + num_test
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        x_init = env.get_init_points(settings["init_num_points"], dim, seed=seed)
        x_init = x_init.to(dtype=dtype, device=device)
        _, y_init = env.evaluate(x_init)
        y_init = y_init.to(dtype=dtype, device=device)

        model.eval()
        for trial in tqdm(range(settings["test_num_trials"])):
            model.x_init = x_init
            model.y_init = (y_init - y_min) / (y_max - y_min)
            
            acqf = UpperConfidenceBound(model, beta=1.0, maximize=False)
            x_cand, _ = optimize_acqf(acqf, bounds=bounds,
                                            q=1, num_restarts=10, raw_samples=512)

            x_cand_transform, y_cand = env.evaluate(x_cand)

            x_init = torch.cat([x_init, x_cand], dim=0)
            y_init = torch.cat([y_init, y_cand], dim=0)

            logger.info(f"[{trial+1}/{settings['test_num_trials']}]\nAction: {x_cand_transform}\nPerformance: {y_cand.item():4f}")

        test_results["actions"] = x_init.cpu().detach().numpy()
        test_results["performance"] = y_init.cpu().detach().numpy().flatten()

        test_results_overall["actions"].append(test_results["actions"])
        test_results_overall["performance"].append(test_results["performance"])

        if args.network == "kt_simulator":
            logger.info(f'Best Performance: {test_results_overall["performance"][-1].max().item():.4f}')
        if args.network.startswith("sumo"):
            logger.info(f'Best Performance: {test_results_overall["performance"][-1].min().item():.4f}')
    
    # Save results
    with open(results_save_path, "wb") as f:
        pickle.dump(test_results_overall, f)
