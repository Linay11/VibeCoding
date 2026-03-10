import os
import time
import warnings
import cplex
import numpy as np
import pandas as pd
from cvxpy import Minimize, Maximize
from tqdm import tqdm
import mlopt.settings as stg
import mlopt
import gurobipy as gp
import online_optimization.binkar.utils as u
from mlopt import utils

warnings.filterwarnings("ignore")
seed = int(time.time()) + os.getpid()
np.random.seed(seed)


def test_solve():
    filedir = '/home/ljj/project/mlopt/online_optimization/neos'
    file = filedir + '/neos-827175.mps'
    sol_file = filedir + '/neos-827175.sol'
    isfile = os.path.isfile(file)
    print(isfile)
    m = gp.read(file)
    m.optimize()
    m.reset()
    problem, b = u.readmpsmodel(file)
    column_names = [f'b{i}' for i in range(len(b))]
    df = pd.DataFrame([b], columns=column_names)
    ml = mlopt.Optimizer(problem, parallel=False)
    res = ml._problem.solve_parametric(df, parallel=False)
    print(res[0]['cost'])


def test_setcover():
    STORAGE_DIR = "/home/ljj/project/mlopt/results/neos/setcover"
    mlopt_model_DIR = "/home/ljj/project/mlopt/results/neos/setcover"
    EXAMPLE_NAME = STORAGE_DIR + '/neos_'

    # Load model
    n_best = 20
    m_mlopt = mlopt.Optimizer.from_file(mlopt_model_DIR + '/neos_model')
    m_mlopt._learner.options['n_best'] = n_best

    m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
    test_problems = m_mlopt.X_train
    test_obj_train = m_mlopt.obj_train
    m_mlopt.cache_factors()
    strategys = m_mlopt.encoding
    correct_list = []
    for problem_idx in tqdm(range(test_problems.shape[0])):
        problem = test_problems.iloc[problem_idx]
        rewards = []
        inputs = []

        m_mlopt._problem.populate(problem)

        results = []
        infeasibilities = []
        costs = []

        sol_time = 0
        for k in range(len(strategys)):
            s = strategys[k]
            res = m_mlopt._problem.solve(strategy=s)
            sol_time += res['time']
            results.append(res)
            infeasibilities.append(res['infeasibility'])
            costs.append(res['cost'])

        infeasibilities = np.array(infeasibilities)
        costs = np.array(costs)
        feasible_indices = np.where(infeasibilities <= stg.INFEAS_TOL)[0]

        if len(feasible_indices) > 0:
            # 选择成本最优的策略
            if m_mlopt._problem.sense() == Minimize:
                best_idx = feasible_indices[np.argmin(costs[feasible_indices])]
            elif m_mlopt._problem.sense() == Maximize:
                best_idx = feasible_indices[np.argmax(costs[feasible_indices])]
            else:
                raise ValueError('Objective type not understood')
        else:
            # 所有策略都不可行，选择最小不可行性的策略
            best_idx = np.argmin(infeasibilities)

        best_infeasibility = infeasibilities[best_idx]
        best_cost = costs[best_idx]
        target_cost = test_obj_train[problem_idx]
        best_suboptimality = utils.suboptimality(best_cost, target_cost, m_mlopt._problem.sense())
        if best_suboptimality <= 0:
            best_suboptimality = 0

        # 判断次优性和不可行性是否满足约束
        if best_infeasibility <= stg.INFEAS_TOL and np.abs(best_suboptimality) <= stg.SUBOPT_TOL:
            idx_correct = 1
        else:
            idx_correct = 0

        correct_list.append(idx_correct)

    print('test done')


if __name__ == '__main__':
    test_solve()
