import gc
import os
import re
import sys

import cplex
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mlopt.sampling import uniform_sphere_sample


def readmpsmodel(sample_name):
    problem = cplex.Cplex()
    problem.read(sample_name)
    #problem.read("../../lp/error_new_model_1026_pre.lp")
    # 转换
    num_variables = problem.variables.get_num()
    num_constraints = problem.linear_constraints.get_num()
    variable_types = problem.variables.get_types()
    int_num = 0
    for i in range(num_variables):
        if variable_types[i] == 'B':
            int_num = int_num + 1

    x_c = cp.Variable(num_variables - int_num, name='x')
    x_i = cp.Variable(int_num, integer=True,  name='x_I')
    x = cp.hstack([x_c, x_i])

    # Get constraint coefficients
    constraint_matrix = []  # 约束变量矩阵
    constraint_vector = []  # 约束系数矩阵
    constraint_senses = []
    constraint_rhs = []
    for i in range(num_constraints):
        row = problem.linear_constraints.get_rows(i)
        constraint_matrix.append([row.ind[j] for j in range(len(np.asarray(row.unpack()[0])))])
        constraint_vector.append([row.val[j] for j in range(len(np.asarray(row.unpack()[0])))])
        constraint_senses.append(problem.linear_constraints.get_senses(i))
        constraint_rhs.append(problem.linear_constraints.get_rhs(i))

    num_b = sum(1 for s in constraint_senses if s == 'L')

    b = []
    for i in range(num_b):
        #c.append(cp.Parameter(nonneg=True,name='c'+str(i)))
        b.append(cp.Parameter(name='b' + str(i)))
    k = 0
    constraints = []
    for i in range(num_constraints):
        arr = np.zeros(num_variables)
        arr[constraint_matrix[i]] = constraint_vector[i]
        arr_sparse = csr_matrix(arr)

        # pattern = r"inv_"
        # match = re.search(pattern, problem.linear_constraints.get_names(i))
        match = False

        if constraint_senses[i] == 'E':
            if k < num_b:
                if match:
                    #c[k] = cp.Parameter(nonneg=True)
                    b[k].value = constraint_rhs[i]
                    constraints.append(arr_sparse @ x == b[k])
                    k = k + 1
                else:
                    constraints.append(arr_sparse @ x == constraint_rhs[i])

            else:
                constraints.append(arr_sparse @ x == constraint_rhs[i])

        elif constraint_senses[i] == 'L':
            if k < num_b:
                if True:
                    b[k].value = constraint_rhs[i]
                    constraints.append(arr_sparse @ x <= b[k])
                    k = k + 1
                else:
                    constraints.append(arr_sparse @ x <= constraint_rhs[i])
            else:
                constraints.append(arr_sparse @ x <= constraint_rhs[i])
        elif constraint_senses[i] == 'G':
            if k < num_b:
                if match:
                    #c[k] = cp.Parameter(nonneg=True)
                    b[k].value = constraint_rhs[i]
                    constraints.append(arr_sparse @ x >= b[k])
                    k = k + 1
                else:
                    constraints.append(arr_sparse @ x >= constraint_rhs[i])

            else:
                constraints.append(arr_sparse @ x >= constraint_rhs[i])

    del constraint_matrix
    del constraint_vector
    del constraint_senses
    del constraint_rhs
    gc.collect()

    arr_bounds = np.eye(num_variables, dtype='float16')

    arr_up = np.zeros(num_variables, dtype='float16')
    arr_low = np.zeros(num_variables, dtype='float16')

    for i in range(num_variables):
        arr_low[i] = problem.variables.get_lower_bounds(i)
        arr_up[i] = problem.variables.get_upper_bounds(i)

    arr_bounds_csr = csr_matrix(arr_bounds)
    del arr_bounds
    gc.collect()

    constraints.append(arr_bounds_csr @ x <= arr_up)
    del arr_up
    gc.collect()
    constraints.append(arr_bounds_csr @ x >= arr_low)

    del arr_bounds_csr
    del arr_low
    gc.collect()

    arrr = np.zeros(num_variables)

    for i in range(len(problem.objective.get_linear())):
        arrr[i] = problem.objective.get_linear(i)

    object_function = cp.Minimize(arrr @ x + problem.objective.get_offset())
    final = cp.Problem(object_function, constraints)

    del object_function
    del constraints
    gc.collect()

    # data = {'c' + str(i): v.value for i, v in enumerate(c)}
    # df = pd.DataFrame(data, index=[0])
    # df.to_csv("../../lp/parameters_small_0_1000.csv", index=False)

    return final


def sim_data(n_all, file_name):
    # df = pd.DataFrame(
    #         {"c0": [2063001.0], "c1": [632174.0], "c2": [379371.0], "c3": [293202.0], "c4": [6723.0], "c5": [3647757.0],
    #          "c6": [4951.0], "c7": [51188.0], "c8": [2222498.0], "c9": [932555.0], "c10": [151.0], "c11": [62062.0]})
    #df = pd.read_csv("../../lp/parameters_error_1026_2_1000.csv")
    #df = pd.read_csv("../../lp/parameters_c1000.csv")
    #df = pd.read_csv("../../lp/parameters_small_0_1000.csv")
    df = pd.read_csv(file_name)
    #df_samples= sample_around_points(df,n_total=10,radius={"c0": 100000,"c1": 10000,"c2": 10000,"c3": 10000,"c4": 100,"c5": 100000,"c6": 100,"c7": 1000,"c8": 100000,"c9": 10000,"c10": 10,"c11": 1000})
    df_samples = sample_around_points(df, n_total=n_all,
                                      radius={})
    #df_samples.to_csv("../../lp/parameters_error_1026_2_1000_samples.csv", index=False)
    return df_samples


def sample_around_points(df,
                         n_total=8000,
                         radius={}):
    """
    Sample around points provided in the dataframe for a total of
    n_total points. We sample each parameter using a uniform
    distribution over a ball centered at the point in df row.
    """
    np.random.seed(0)
    n_samples_per_point = np.round(n_total / len(df), decimals=0).astype(int)

    df_samples = pd.DataFrame()

    for idx, row in df.iterrows():
        df_row = pd.DataFrame()

        # For each column sample points and create series
        for col in df.columns:

            norm_val = np.linalg.norm(row[col])
            if norm_val < 1e-4:
                norm_val = 1.

            if col in radius:
                rad = radius[col] * norm_val
            else:
                rad = 1e-01 * norm_val

            samples = uniform_sphere_sample(row[col], rad,
                                            n=n_samples_per_point)

            samples = np.maximum(samples, 0)

            if len(samples[0]) == 1:
                # Flatten list
                samples = [item for sublist in samples for item in sublist]

            df_row[col] = list(samples)

        df_samples = df_samples.append(df_row)

    return df_samples


if __name__ == '__main__':
    filedir = '/home/ljj/project/mlopt/online_optimization/binkar'
    file = filedir + '/binkar10_1.mps'
    sol_file = filedir + '/binkar10_1.sol'
    problem = readmpsmodel(file)
