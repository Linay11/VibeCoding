# Needed for slurm
import os
import sys

sys.path.append(os.getcwd())
import online_optimization.control.utils as u
import numpy as np
import mlopt
from mlopt.optimizer import Optimizer
import pickle
import argparse
import os
import pandas as pd
import online_optimization.settings as st
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)

if __name__ == '__main__':

    desc = 'Online Control Example'
    testbuild = True
    trainbuild = False
    cluster = True
    parser = argparse.ArgumentParser(description=desc)

    T = 40
    parser.add_argument('--horizon', type=int, default=T, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon
    if cluster:
        STORAGE_DIR = st.ROOT_DIR + "results/control/setcover"
        EXAMPLE_NAME = STORAGE_DIR + '/control_%d_setcover' % T_horizon
    else:
        STORAGE_DIR =st.ROOT_DIR + "results/control/test"
        EXAMPLE_NAME = STORAGE_DIR + '/control_%d_' % T_horizon

    # Problem data
    n_traj = 10000  # Trajectory sampling to get points
    tau = 1.0
    n_train = 10000

    n_test = 1000

    seed_train = 2
    seed_test = 1
    seed_valid = 3

    print(desc, " N = %d\n" % T_horizon)

    # Get trajectory
    P_load_train = u.P_load_profile(n_traj, seed=seed_train)

    # Create simulation data
    init_data_train = {'E': [7.7],
                       'z': [0.],
                       's': [0.],
                       'P': [],
                       'past_d': [np.zeros(T_horizon)],
                       'P_load': [P_load_train[:T_horizon]],
                       'sol': []}

    # Get trajectory
    P_load_test = u.P_load_profile(n_test, seed=seed_test)

    # Create simulation data
    init_data_test = {'E': [7.7],
                      'z': [0.],
                      's': [0.],
                      'P': [],
                      'past_d': [np.zeros(T_horizon)],
                      'P_load': [P_load_test[:T_horizon]],
                      'sol': []}

    # Define problem
    problem, cost_function_data = u.control_problem(T_horizon, tau=tau)

    # Create mlopt problem
    m_mlopt = Optimizer(problem,
                              parallel=True)

    # Check if learning data already there
    if trainbuild and not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):
        print("Get learning data by simulating closed loop———— Train")
        sim_data = u.simulate_loop(problem, init_data_train,
                                   u.basic_loop_solve,
                                   P_load_train,
                                   n_traj,
                                   T_horizon)

        # lp_base_file_name = '/home/ljj/project/mlopt/results/control/instance_%d' % T_horizon
        # if not os.path.isdir(lp_base_file_name):
        #     os.makedirs(lp_base_file_name)
        # u.save_multiple_lp_files(sim_data, problem, lp_base_file_name, '/setcover_control_%d.lp')

        # Store simulation data as parameter values (avoid sol parameter)
        df = u.sim_data_to_params(sim_data)

        # Sample over balls around all the parameters
        df_train = u.sample_around_points(df,
                                          radius={'z_init': .4,  # .2,
                                                  #  's_init': .6,  # .2,
                                                  'P_load': 0.001,  # 0.01
                                                  },
                                          n_total=n_train)
        # Get samples
        m_mlopt.get_samples(df_train,
                            parallel=True,
                            filter_strategies=False)  # Filter strategies after saving
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                                   delete_existing=True)


    if testbuild and not os.path.isfile(EXAMPLE_NAME + 'test_data.pkl'):
        print("Get learning data by simulating closed loop ———— Test")
        sim_data = u.simulate_loop(problem, init_data_test,
                                   u.basic_loop_solve,
                                   P_load_test,
                                   n_test,
                                   T_horizon)

        # Store simulation data as parameter values (avoid sol parameter)
        df = u.sim_data_to_params(sim_data)

        # Sample over balls around all the parameters
        df_test = u.sample_around_points(df,
                                         radius={'z_init': .4,  # .2,
                                                 #  's_init': .6,  # .2,
                                                 'P_load': 0.001,  # 0.01
                                                 },
                                         n_total=n_test)
        # Get samples
        m_mlopt.get_samples(df_test,
                            parallel=True,
                            filter_strategies=False)  # Filter strategies after saving
        m_mlopt.save_training_data(EXAMPLE_NAME + 'test_data.pkl',
                                   delete_existing=True)

    else:
        print("Loading data from file")

    if trainbuild:
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')
        # Filter strategies and resave
        if cluster:
            print('using Cluster Filter!')
            m_mlopt.cluster_filter(parallel=True)
        else:
            m_mlopt.filter_strategies(parallel=True)
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
                                   delete_existing=True)

    # # Learn optimizer
    # m_mlopt.train(learner=mlopt.PYTORCH,
    #               n_best=10,
    #               filter_strategies=False,  # Do not filter strategies again
    #               #  n_train_trials=2,
    #               parallel=True)
    #
    # # Save model
    # m_mlopt.save(EXAMPLE_NAME + 'model', delete_existing=True)
