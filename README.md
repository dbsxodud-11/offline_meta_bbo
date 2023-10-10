# An Offline Meta Black-box Optimization Framework for Adaptive Design of Traffic Light Scheme on Urban Road Networks

### Environment Setup

You should install SUMO, an urban traffic simulator. Please follow instructions in [here](https://sumo.dlr.de/docs/Downloads.php)

We strongly recommend to use conda environment. We can install required libraries using `requirements.txt`
```
conda create -n tsc python=3.8 -y
conda activate tsc
pip install -r requirements.txt
```

### Main Experiments

Our method consists of three main steps: Offline Data Collection, Meta Training, and Online Adaptation.

- Offline Data Collection
    To collect offline data, you should run the following command. You can set `num_worker` to accelerate the data collection process via parallelization.

    ```
    python collect_data.py --network 2by2 --scheme comb
    ```
    We provide script file for generating offline data across all networks and schemes in `scripts/collect_data.sh`


- Meta Training
    After collecting dataset, you should run the following command to train our ANP model. Model will be saved in `results/<network>/<scheme>/anp/<exp_id>/ckpt.tar`.

    ```
    python meta_train.py --network 2by2 --scheme comb --model anp --exp_id trial1
    ```
    We provide script file for meta-training ANP across all networks and schemes in `scripts/meta_train.sh`

- Online Adaptation
    Finally, we employ Bayesian optimization with the trained ANP as a surrogate model to find optimal scheme for traffic lights with unseen traffic patterns. 
    You should run the following command to conduct online adaptation.

    ```
    python meta_test.py --network 2by2 --scheme comb --model anp --exp_id trial1 --scenario_id 0
    ```
    We provide script file for online adaptation across all networks and schemes in `scripts/meta_test.sh`
