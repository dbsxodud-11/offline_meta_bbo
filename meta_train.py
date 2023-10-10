import os
from os.path import join as pjoin
import random
import argparse

import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from attrdict import AttrDict
from collections import defaultdict

from data.traffic import DataSampler
from utils.log import get_logger
from utils.misc import load_module


if __name__ == "__main__":

    # Argument Passing
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="2by2")
    parser.add_argument("--scheme", type=str, default="comb")
    parser.add_argument("--model", type=str, default="anp")
    parser.add_argument("--exp_id", type=str, default="trial1")
    parser.add_argument("--root", type=str, default='.')
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    network = args.network
    scheme = args.scheme
    exp_id = args.exp_id
    seed = args.seed
        
    root = pjoin(args.root, "results", network, scheme, exp_id)
    if not os.path.isdir(root):
        os.makedirs(root)

    # Set Path for model checkpoint, settings, arguments, and logger
    model_ckpt_save_path = pjoin(root, "ckpt.tar")
    model_args_save_path = pjoin(root, "model.yaml")
    settings_save_path = pjoin(root, "settings.yaml")
    log_path = pjoin(root, "execution.log")
    logger = get_logger(log_path)

    # Set Hyperparameters
    settings = yaml.load(open(pjoin(args.root, "config", network, "train_settings.yaml"), "r"), Loader=yaml.SafeLoader)

    # Set Seed for Reproduction
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Save Model and Setting Arguments
    model_args = yaml.load(open(pjoin(args.root, "config", network, f"{args.model}_{scheme}.yaml"), "r"), Loader=yaml.SafeLoader)

    with open(model_args_save_path, 'w') as f:
        yaml.dump(model_args, f)

    with open(settings_save_path, 'w') as f:
        yaml.dump(settings, f)

    # Set Task
    sampler = DataSampler(data_path=pjoin(args.root, "data", network, f"traffic_data_{scheme}.pkl"), dtype=dtype, device=device)
    sampler_val = DataSampler(data_path=pjoin(args.root, "data", network, f"traffic_data_valid_{scheme}.pkl"), dtype=dtype, device=device)

    # Construct Model with arguments written in config file
    model_cls = getattr(load_module(pjoin(args.root, "models", f"{args.model}.py")), args.model.upper())
    with open(pjoin(args.root, "config", network, f"{args.model}_{scheme}.yaml"), "r") as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).to(dtype=dtype, device=device)

    # Using Adam Optimizer and Cosine Annealing Learning rate Scheduler for robust convergence
    optimizer = optim.Adam(model.parameters(), lr=settings["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["train_num_epochs"])

    # Check number of parameters in model
    logger.info(f"Total Number of Parameters: {sum(p.numel() for p in model.parameters())}")

    # Training
    loss_mean = 0.0
    val_loss_max = -1000
    for epoch in tqdm(range(1, settings["train_num_epochs"]+1)):
        # 1. Sample batch data from sampler, which contains (input, output) pair obtained from previous simulation
        batch = sampler.sample(settings["train_batch_size"], settings["train_max_num_points"])
        
        # 2. Forward Model, output will be a probability distribution for y values
        model.train()
        out = model(batch.x_context, batch.y_context, batch.x, batch.y,
                    num_samples=settings["train_num_samples"])

        # 3. Calculate Loss
        loss = model.calculate_loss(out, batch.y, num_samples=settings["train_num_samples"])

        # 4. Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 5. Logging
        loss_mean += loss.item()
        if epoch % settings["print_freq"] == 0:
            logger.info(f'[{epoch}/{settings["train_num_epochs"]}]\tmodel: {args.model}\tloss: {loss_mean / settings["print_freq"]:.4f}')
            loss_mean = 0.0

        # Validation
        if epoch % settings["val_freq"] == 0:
            val_results = defaultdict(list)
            for _ in tqdm(range(settings["val_num_epochs"])):
                # 1. Sample batch data from sampler, which contains (input, output) pair obtained from previous simulation
                val_batch = sampler_val.sample(settings["val_batch_size"], settings["val_max_num_points"])
                
                # 2. Forward Model, output will be a probability distribution for y values
                model.eval()
                val_out = model(val_batch.x_context, val_batch.y_context, val_batch.x,
                                num_samples=settings["val_num_samples"])
                
                # 3. Evaluate Model, output will be log likelihood for context and target points. Higher values implies the
                # model accurately predict the y values
                val_result = model.evaluate(val_out, val_batch.y_context, val_batch.y,
                                            num_samples=settings["val_num_samples"])

                val_results["context_ll"].append(val_result["context_ll"])
                val_results["target_ll"].append(val_result["target_ll"])
                
            logger.info(f'context_ll: {np.array(val_results["context_ll"]).mean():.4f}')
            logger.info(f'target_ll: {np.array(val_result["target_ll"]).mean():.4f}')

            if np.array(val_result["target_ll"]).mean() > val_loss_max:
                val_loss_max = np.array(val_result["target_ll"]).mean()

                ckpt = AttrDict()
                ckpt.model = model.state_dict()
                ckpt.optimizer = optimizer.state_dict()
                ckpt.scheduler = scheduler.state_dict()
                torch.save(ckpt, model_ckpt_save_path)
                print('save model !')
