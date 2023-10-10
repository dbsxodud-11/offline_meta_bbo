import os
import random
from os.path import join as pjoin
from copy import deepcopy

import torch
from torch.quasirandom import SobolEngine
from tqdm import tqdm
from collections import defaultdict

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort
import torch.multiprocessing as mp

class SumoEnv():
    def __init__(self, network, scheme, scenario, root, run_type='train', visualize=False):
        # SUMO Configuration Setting
        if visualize:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        sumocfg_path = pjoin(root, "env", network, f"{run_type}_{scenario}.sumocfg")
        init_tllogic_path = pjoin(root, "env", network, f"{network}.tll.xml")

        self.sumoCmd = [sumoBinary, "-c", sumocfg_path]
        self.sumoCmd += ["-a", init_tllogic_path]
        self.sumoCmd += ['--time-to-teleport', '600'] # long teleport for safety
        self.sumoCmd += ['--no-warnings', 'True']
        self.sumoCmd += ['--duration-log.disable', 'True']
        self.sumoCmd += ['--no-step-log', 'True']

        self.port = getFreeSocketPort()

        self.time = 0
        if network == "manhattan-large":
            self.decision_time = 900
        else:
            self.decision_time = 1800
        self.eval_time = 60
        self.yellow_time = 3

        self.min_time = 30
        self.cycle_time = 180
        self.scheme = scheme
        
        if self.scheme == "comb":
            self.input_dim = 4
        elif self.scheme == "timing":
            self.input_dim = 4
        else:
            raise NotImplementedError
        
        if network == "2by2":
            self.n_intersections = 4
        elif network == "3by3":
            self.n_intersections = 9
        elif network == "4by4":
            self.n_intersections = 16
        elif network == "5by5":
            self.n_intersections = 25
        elif network == "hangzhou":
            self.n_intersections = 16
        elif network == "manhattan":
            self.n_intersections = 48
        elif network == "manhattan-large":
            self.n_intersections = 196  

    def apply_action(self, x):
        traci.start(self.sumoCmd, port=self.port)

        if self.scheme == "comb":
            for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
                program_ID = torch.argmax(x[i]).item() + 1
                traci.trafficlight.setProgram(intersection_ID, program_ID)     
        elif self.scheme == "timing":
            x_dict = {}
            for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
                x_dict[intersection_ID] = x[i].tolist()
                current_logic = traci.trafficlight.getAllProgramLogics(intersection_ID)[1]
                next_logic = deepcopy(current_logic)
                
                current_phases = current_logic.getPhases()
                next_phases = []
                for j, phase in enumerate(current_phases):
                    if j % 2 == 0:
                        phase.duration = int(x[i][j//2].item() * (self.cycle_time - (self.min_time + self.yellow_time) * self.input_dim) + self.min_time)
                        phase.minDur = phase.duration
                        phase.maxDur = phase.duration
                    next_phases.append(phase)
                next_phases = tuple(next_phases)

                next_logic.phases = next_phases
                traci.trafficlight.setProgramLogic(intersection_ID, next_logic)
                traci.trafficlight.setProgram(intersection_ID, 1)

        y = torch.zeros(self.n_intersections, self.decision_time // self.eval_time).to(x.device)
        y_analysis = defaultdict(list)
        for t in range(self.decision_time):
            traci.simulationStep()
            if (t+1) % self.eval_time == 0:
                for i, intersection_ID in enumerate(traci.trafficlight.getIDList()) :
                    for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                        y[i][t // self.eval_time] += traci.lane.getLastStepHaltingNumber(lane)

                for edgeID in traci.edge.getIDList():
                    y_analysis[edgeID].append(traci.edge.getLastStepHaltingNumber(edgeID))
        y_analysis = {key: sum(value) / 30 for key, value in y_analysis.items()}

        traci.close()
        return y.mean(dim=-1)

    def get_constraints(self, dtype, device):
        equality_constraints = []
        total_input_dim = self.input_dim * self.n_intersections

        prev_input_dim = 0
        for _ in range(self.n_intersections):
            indices = torch.arange(prev_input_dim, prev_input_dim+self.input_dim).to(device)
            coefficients = torch.ones(self.input_dim).to(device)
            equality_constraints.append((indices, coefficients, 1.0))
            prev_input_dim += self.input_dim

        bounds = torch.stack([torch.zeros(total_input_dim), torch.ones(total_input_dim)], dim=0).to(dtype=dtype, device=device)
        return total_input_dim, bounds, equality_constraints

    def get_init_points(self, init_num_points, dim, seed=42):
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        x = sobol.draw(init_num_points)
        return x

    def input_transform(self, x):
        return x.view(-1, self.n_intersections, self.input_dim).softmax(dim=-1)

    def evaluate(self, xs, aggregate=True):
        xs = self.input_transform(xs)
        if aggregate:
            ys = [self.apply_action(x).sum(dim=-1, keepdim=True) for x in xs]
        else:
            ys = []
            for i in tqdm(range(len(xs))):
                ys.append(self.apply_action(xs[i]))

        if len(xs) == 1:
            xs = xs.squeeze(0).tolist()
        ys = torch.stack(ys)
        return xs, ys

def env_evaluation(config):
    env, xs = config
    ys = []
    for i in range(len(xs)):
        ys.append(env.apply_action(xs[i]))
    return xs, torch.stack(ys)


def parallelize_sumo(args, scenario, settings, number_of_worker, run_type):
    envs = [SumoEnv(args.network, args.scheme, scenario, args.root, run_type=run_type) for _ in range(number_of_worker)]

    seed = random.randint(0, 1e6)
    xs_orig = envs[0].get_init_points(settings["num_samples_per_scenario"], settings[f"raw_input_dim"], seed=seed)
    xs = envs[0].input_transform(xs_orig)
    xs = xs.reshape((number_of_worker, -1, *xs.shape[-2:]))
    xs = [xs[i] for i in range(number_of_worker)]

    multi_worker = mp.Pool(number_of_worker)
    multi_worker_result = multi_worker.map(env_evaluation, zip(envs, xs))
    ys = []
    for result in multi_worker_result:
        _, y = result
        ys.append(y)
    ys = torch.cat(ys, dim=0)

    return xs_orig, ys
