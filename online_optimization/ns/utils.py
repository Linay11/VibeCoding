import gc
import os
import re
import sys
from tqdm import tqdm
import cplex
import cvxpy as cp
import gurobi as gp
import mlopt.error as e
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mlopt.sampling import uniform_sphere_sample


def readmpsmodel(sample_name):
    problem = cplex.Cplex()
    problem.read(sample_name)
    # problem.read("../../lp/error_new_model_1026_pre.lp")
    # 转换
    num_variables = problem.variables.get_num()
    num_constraints = problem.linear_constraints.get_num()
    variable_types = problem.variables.get_types()
    int_num = 0
    for i in range(num_variables):
        if variable_types[i] == 'B':
            int_num = int_num + 1

    x_c = cp.Variable(num_variables - int_num, name='x')
    x_i = cp.Variable(int_num, integer=True, name='x_I')
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

    num_b = 100
    ineq_ind = -1
    b = []
    b_center = np.empty(num_b)
    for i in range(num_b):
        # c.append(cp.Parameter(nonneg=True,name='c'+str(i)))
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
                    # c[k] = cp.Parameter(nonneg=True)
                    b[k].value = constraint_rhs[i]
                    b_center[k] = constraint_rhs[i]
                    constraints.append(arr_sparse @ x == b[k])
                    k = k + 1
                else:
                    constraints.append(arr_sparse @ x == constraint_rhs[i])

            else:
                constraints.append(arr_sparse @ x == constraint_rhs[i])

        elif constraint_senses[i] == 'L':
            ineq_ind += 1
            if k < num_b:
                if 576 <= ineq_ind < 676:
                    b[k].value = constraint_rhs[i]
                    b_center[k] = constraint_rhs[i]
                    constraints.append(arr_sparse @ x <= b[k])
                    k = k + 1
                else:
                    constraints.append(arr_sparse @ x <= constraint_rhs[i])
            else:
                constraints.append(arr_sparse @ x <= constraint_rhs[i])
        elif constraint_senses[i] == 'G':
            ineq_ind += 1
            if k < num_b:
                if 576 <= ineq_ind < 676:
                    # c[k] = cp.Parameter(nonneg=True)
                    b[k].value = constraint_rhs[i]
                    b_center[k] = constraint_rhs[i]
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
    cvxpy_problem = cp.Problem(object_function, constraints)

    del object_function
    del constraints
    gc.collect()

    # data = {'c' + str(i): v.value for i, v in enumerate(c)}
    # df = pd.DataFrame(data, index=[0])
    # df.to_csv("../../lp/parameters_small_0_1000.csv", index=False)

    return cvxpy_problem, b_center


def sample_around_points(b, r,
                         n_total=8000):
    """
    input:
        b:ndarray
        r: radius
    return:
        DataFrame
    """
    np.random.seed(0)
    column_names = [f'b{i}' for i in range(len(b))]
    df_samples = pd.DataFrame(columns=column_names)
    num_b = len(b)
    for i in tqdm(range(n_total)):
        b_new = sample(r, num_b, b)
        df_row = pd.DataFrame([b_new], columns=column_names)
        df_samples = pd.concat([df_samples, df_row], ignore_index=True)

    return df_samples


def sample(r, ndim, old):
    dn = np.random.randn(ndim + 2, 1)
    norms_d = np.linalg.norm(dn)
    dn_normalized = dn / norms_d
    new = (dn_normalized[:ndim].flatten() * r) + old
    return new


def modify_constraints_rhs(model, new_rhs, start_idx, end_idx):
    constraints = model.getConstrs()
    k = 0
    for idx, constr in enumerate(constraints):
        if constr.Sense in [gp.GRB.LESS_EQUAL, gp.GRB.GREATER_EQUAL]:
            if start_idx <= k <= end_idx:
                model.setAttr("RHS", constr, new_rhs[k - start_idx])
            k += 1


def modify_objective(model, new_coeffs, parameter_indices, model_sense):
    old_coeffs = model.getAttr("Obj")
    new_obj = old_coeffs[:]  # 拷贝原始系数
    variables = model.getVars()
    for idx, param_idx in enumerate(parameter_indices):
        new_obj[param_idx] = new_coeffs[idx]  # 更新目标系数
    model.setObjective(gp.LinExpr(new_obj, variables), model_sense)


def get_instance(sample_name, n_instances, r, ptype='binkar'):
    mode = "test"
    instace_dir = f'/home/ljj/project/predict_and_search/instance/{mode}/{ptype}/'
    if not os.path.exists(instace_dir):
        os.makedirs(instace_dir)
    model = gp.read(sample_name)
    variables = model.getVars()
    start = 576
    end = 675
    constraints = model.getConstrs()
    original_params = []
    k = 0
    for idx, constr in enumerate(constraints):
        if constr.Sense in [gp.GRB.LESS_EQUAL, gp.GRB.GREATER_EQUAL]:
            if start <= k <= end:
                rhs = constr.RHS
                original_params.append(rhs)
            k += 1

    ndim = end - start + 1
    model_sense = gp.GRB.MINIMIZE if model.getAttr("ModelSense") == 1 else gp.GRB.MAXIMIZE

    if mode == "test":
        new_params_list = read_new_para(f"/home/ljj/project/mlopt/online_optimization/{ptype}/{ptype}_{r}_test_data.pkl")
    elif mode == "train":
        new_params_list = read_new_para(f"/home/ljj/project/mlopt/online_optimization/{ptype}/{ptype}_{r}_data.pkl")
    print('generating instance....')
    n_instances = len(new_params_list)

    print('generating instance....')
    for i in tqdm(range(n_instances)):
        instance_file = instace_dir + f"{ptype}_{i}.lp"
        # new_params = sample(r, ndim, original_params)
        new_params = new_params_list[i]
        # modify_objective(model, new_params, model_sense)
        modify_constraints_rhs(model, new_params, start, end)
        try:
            model.write(instance_file)
        except Exception as e:
            print(f"Error saving instance {i}: {str(e)}")

def read_new_para(file_name):
    if not os.path.isfile(file_name):
        e.value_error("File %s does not exist." % file_name)
    data_dict = pd.read_pickle(file_name)
    params_dict = data_dict['X_train']
    new_para_list = np.array(params_dict).tolist()
    return new_para_list

print("start generating")
ptype = "ns"
filedir = f'/home/ljj/project/mlopt/online_optimization/{ptype}'
file = filedir + '/ns1830653.mps'
get_instance(file, 100, r=0.02, ptype=ptype)
