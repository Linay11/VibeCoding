# Needed for slurm
import os
import sys

sys.path.append(os.getcwd())
import online_optimization.beasley.utils as u
import numpy as np
import mlopt
import pickle
import argparse
import os
import pandas as pd

import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)

if __name__ == '__main__':

    desc = 'Online beasleyC3 Example'
    testbuild = True
    trainbuild = True
    isfilter = False
    cluster = True
    parser = argparse.ArgumentParser(description=desc)
    r = 1
    ptype = "beasley"

    if cluster:
        STORAGE_DIR = f"/home/ljj/project/mlopt/results/{ptype}/setcover"
        EXAMPLE_NAME = STORAGE_DIR + f'/{ptype}_{r}_'
    else:
        STORAGE_DIR = f"/home/ljj/project/mlopt/results/{ptype}/RLHF"
        EXAMPLE_NAME = STORAGE_DIR + f'/{ptype}_{r}_'

    # Problem data
    n_train = 2000
    n_test = 100

    seed_train = 2
    seed_test = 1
    seed_valid = 3
    filedir = f'/home/ljj/project/mlopt/online_optimization/{ptype}'
    file = filedir + '/beasleyC3.mps'

    problem, b = u.readmpsmodel(file)

    m_mlopt = mlopt.Optimizer(problem,
                              parallel=True)

    # Check if learning data already there
    if trainbuild and not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):
        # 训练集
        print("Get learning data by simulating closed loop ———— Train")

        df_train = u.sample_around_points(b, r, n_total=n_train)

        m_mlopt.get_samples(df_train,
                            parallel=True,
                            filter_strategies=False)  # Filter strategies after saving
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                                   delete_existing=True)
    else:
        print("Loading data from file")
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')

    if testbuild and not os.path.isfile(EXAMPLE_NAME + 'test_data.pkl'):
        # 测试集
        print("Get learning data by simulating closed loop ———— Test")

        df_test = u.sample_around_points(b, r, n_total=n_test)

        m_mlopt.get_samples(df_test,
                            parallel=True,
                            filter_strategies=False)  # Filter strategies after saving
        m_mlopt.save_training_data(EXAMPLE_NAME + 'test_data.pkl',
                                   delete_existing=True)

    if trainbuild and not os.path.isfile(EXAMPLE_NAME + 'data_filtered.pkl'):
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')
        # Filter strategies and resave
        if cluster and isfilter:
            print('using Cluster Filter!')
            m_mlopt.cluster_filter(parallel=True)
        elif isfilter:
            m_mlopt.filter_strategies(parallel=True)
        else:
            print('train with no Filter!')
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
                                   delete_existing=True)
    else:
        print("Loading filter data from file")
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')

    # Learn optimizer
    m_mlopt.train(learner=mlopt.PYTORCH,
                  n_best=10,
                  filter_strategies=False,  # Do not filter strategies again
                  #  n_train_trials=2,
                  parallel=True)

    # Save model
    m_mlopt.save(EXAMPLE_NAME + 'model', delete_existing=True)
