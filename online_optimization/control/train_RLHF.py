# Needed for slurm
import os
import sys
import time
import uuid

import math
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
import mlopt.settings as stg

sys.path.append(os.getcwd())
import online_optimization.control.utils as u
import numpy as np
import json
import mlopt
import pickle
import argparse
import os
import pandas as pd
from mlopt import utils
from cvxpy import Minimize, Maximize
import warnings

warnings.filterwarnings("ignore")
seed = int(time.time()) + os.getpid()
np.random.seed(seed)


def strategy2array(s):
    """Convert strategy to array"""
    return np.concatenate([s.tight_constraints, s.int_vars])


def problem2array(p):
    E_init = np.array([p['E_init']])
    z_init = np.array([p['z_init']])
    s_init = np.array([p['s_init']])
    past_d = p['past_d']
    P_load = p['P_load']

    # 输出转换后的NumPy数组
    return np.concatenate((E_init, z_init, s_init, past_d, P_load))


def pandas2array(X):
    """
        Unroll dataframe elements to construct 2d array in case of
        cells containing tuples.
        """

    if isinstance(X, np.ndarray):
        # Already numpy array. Return it.
        return X
    else:
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).transpose()
        # get number of datapoints
        n_data = len(X)

        x_temp_list = []
        for i in range(n_data):
            x_temp_list.append(
                np.concatenate([np.atleast_1d(v).flatten()
                                for v in X.iloc[i].values])
            )
        X_new = np.vstack(x_temp_list)

    return X_new


def get_score(res, suboptim):
    if suboptim <= 0:
        suboptim = 0
    score = (np.abs(suboptim) + res['infeasibility'])
    return score


# 这个函数有点问题，c1和c2应该是一样的，传入的时候记得弄成一样的就行
def compare_solutions(m, problem, s1, s2, c, j, k, res):
    # 加载问题
    m._problem.populate(problem)

    # 用策略求解问题，得到次优性和不可行性
    if res[j] is None:
        res1 = m._problem.solve(strategy=s1)
        res[j] = res1
    else:
        res1 = res[j]

        # 如果 res[k] 为空，则求解并保存到 res[k]
    if res[k] is None:
        res2 = m._problem.solve(strategy=s2)
        res[k] = res2
    else:
        res2 = res[k]

    subopt1 = utils.suboptimality(res1['cost'], c, m._problem.sense())
    subopt2 = utils.suboptimality(res2['cost'], c, m._problem.sense())

    score1 = get_score(res1, subopt1)
    score2 = get_score(res2, subopt2)

    # 返回
    if np.abs(score2 - score1) < stg.INFEAS_TOL:
        return 'tie'
    elif score1 < score2:
        return '1'
    elif score1 > score2:
        return '2'
    else:
        return 'tie'


# 计算一个策略的分数，可以调权重，而不是简单加起来
def cal_score(m, strategy, target_cost):
    res = m._problem.solve(strategy=strategy)
    subopt = utils.suboptimality(res['cost'], target_cost, m._problem.sense())
    score = get_score(res, subopt)
    return score


# def new_data_generator(m, X_train, encodings, obj_train, num_problems_per_call=10):
#     num_problems = X_train.shape[0]
#     num_strategies = len(encodings)
#     problem_indices = np.random.permutation(num_problems).tolist()
#
#     while True:  # 使得生成器可以无限产生数据
#         current_batch_indices = []
#
#         # 收集足够的问题索引用于这一次调用
#         while len(current_batch_indices) < num_problems_per_call:
#             if not problem_indices:  # 如果问题索引为空，则重新加载和打乱
#                 problem_indices = np.random.permutation(num_problems).tolist()
#             current_batch_indices.append(problem_indices.pop())
#
#         # for i in range(num_problems):
#         #     problem = X_train.iloc[i]
#         #     m._problem.populate(problem)
#         #     s = encodings[m.y_train[i]]
#         #     res = m._problem.solve(strategy=s)
#         #     target_cost = obj_train[i]
#         #     subopt = utils.suboptimality(res['cost'], target_cost, m._problem.sense())
#         #     score = np.abs(subopt) + res['infeasibility']
#         #     print(score)
#
#         for idx in current_batch_indices:
#             problem = X_train.iloc[idx]
#             p = pandas2array(problem)
#             m._problem.populate(problem)
#
#             target_cost = obj_train[idx]
#
#             # 计算所有策略的分数，进行排序
#             scores = [cal_score(m, encodings[j], target_cost) for j in range(num_strategies)]
#             sorted_indices = np.argsort(scores)
#
#             sigma1_list = []
#             sigma2_list = []
#             mu_list = []
#
#             # 生成顺序策略对
#             # for j in range(num_strategies - 1):
#             #     k = j + 1
#             #     s1 = strategy2array(encodings[j])[np.newaxis, :]
#             #     s2 = strategy2array(encodings[k])[np.newaxis, :]
#             #     sigma1 = np.concatenate([p, s1], axis=1)
#             #     sigma2 = np.concatenate([p, s2], axis=1)
#             #
#             #     result = compare_solutions(m, problem, encodings[j], encodings[k], obj_train[j], obj_train[k])
#             #     if result == '1':
#             #         mu = [1, 0]
#             #     elif result == '2':
#             #         mu = [0, 1]
#             #     elif result == 'tie':
#             #         mu = [0.5, 0.5]
#             #     else:
#             #         continue  # 忽略不可比较的情况
#             #
#             #     sigma1_list.append(sigma1)
#             #     sigma2_list.append(sigma2)
#             #     mu_list.append(mu)
#
#             # 生成顺序策略对
#             for j in range(len(sorted_indices) - 1):
#                 s1_idx = sorted_indices[j]
#                 s2_idx = sorted_indices[j + 1]
#
#                 s1 = strategy2array(encodings[s1_idx])[np.newaxis, :]
#                 s2 = strategy2array(encodings[s2_idx])[np.newaxis, :]
#                 sigma1 = np.concatenate([p, s1], axis=1)
#                 sigma2 = np.concatenate([p, s2], axis=1)
#
#                 # 设置mu，始终优先考虑第一个策略
#                 mu = [1, 0]
#
#                 sigma1_list.append(sigma1)
#                 sigma2_list.append(sigma2)
#                 mu_list.append(mu)
#
#             if sigma1_list:
#                 yield np.array(sigma1_list), np.array(sigma2_list), np.array(mu_list)

# 不用管，之前搞策略排序的

# 目前弃用
def new_data_generator(m, X_train, encodings, obj_train, num_problems_per_call=10):
    num_problems = X_train.shape[0]
    num_strategies = len(encodings)
    problem_indices = np.random.permutation(num_problems).tolist()

    while True:  # 使得生成器可以无限产生数据
        current_batch_indices = []

        # 收集足够的问题索引用于这一次调用
        while len(current_batch_indices) < num_problems_per_call:
            if not problem_indices:  # 如果问题索引为空，则重新加载和打乱
                problem_indices = np.random.permutation(num_problems).tolist()
            current_batch_indices.append(problem_indices.pop())

        sigma1_total_list = []
        sigma2_total_list = []
        mu_total_list = []

        for idx in current_batch_indices:
            problem = X_train.iloc[idx]
            p = pandas2array(problem)
            target_cost = obj_train[idx]

            # 计算所有策略的分数并排序
            scores = [cal_score(m, encodings[j], target_cost) for j in range(num_strategies)]
            sorted_indices = np.argsort(scores)  # 分数从低到高排序的索引

            sigma1_list = []
            sigma2_list = []
            mu_list = []

            # 生成顺序策略对
            for j in range(len(sorted_indices) - 1):
                s1_idx = sorted_indices[j]
                s2_idx = sorted_indices[j + 1]

                s1 = strategy2array(encodings[s1_idx])[np.newaxis, :]
                s2 = strategy2array(encodings[s2_idx])[np.newaxis, :]
                sigma1 = np.concatenate([p, s1], axis=1)
                sigma2 = np.concatenate([p, s2], axis=1)

                score1 = scores[s1_idx]
                score2 = scores[s2_idx]

                # 根据分数差距决定mu
                if np.abs(score2 - score1) < stg.INFEAS_TOL:
                    mu = [0.5, 0.5]
                else:
                    mu = [1, 0] if score1 < score2 else [0, 1]

                sigma1_list.append(sigma1)
                sigma2_list.append(sigma2)
                mu_list.append(mu)

            sigma1_total_list.extend(sigma1_list)
            sigma2_total_list.extend(sigma2_list)
            mu_total_list.extend(mu_list)

        # 打乱所有列表
        combined = list(zip(sigma1_total_list, sigma2_total_list, mu_total_list))
        np.random.shuffle(combined)
        sigma1_total_list, sigma2_total_list, mu_total_list = zip(*combined)

        if sigma1_total_list:
            yield np.array(sigma1_total_list), np.array(sigma2_total_list), np.array(mu_total_list)


# 验证用的样本生成
def valid_data_generator(m, X_val, encodings, obj_val):
    num_problems = X_val.shape[0]
    num_strategies = len(encodings)

    while True:  # 使得生成器可以无限产生数据
        # 使用所有问题

        problem_indices = np.arange(num_problems)

        sigma1_list = []
        sigma2_list = []
        mu_list = []

        for idx in problem_indices:
            problem = X_val.iloc[idx]
            p = pandas2array(problem)
            target_cost = obj_val[idx]

            # 确保策略数量大于1以选择两个不同策略
            if num_strategies > 1:
                j, k = np.random.choice(num_strategies, 2, replace=False)
                s1 = strategy2array(encodings[j])[np.newaxis, :]
                s2 = strategy2array(encodings[k])[np.newaxis, :]

                sigma1 = np.concatenate([p, s1], axis=1)
                sigma2 = np.concatenate([p, s2], axis=1)

                result = compare_solutions(m, problem, encodings[j], encodings[k], target_cost, target_cost)

                if result == '1':
                    mu = [1, 0]
                elif result == '2':
                    mu = [0, 1]
                elif result == 'tie':
                    mu = [0.5, 0.5]
                else:
                    continue  # 忽略不可比较的情况

                sigma1_list.append(sigma1)
                sigma2_list.append(sigma2)
                mu_list.append(mu)

        if sigma1_list:
            yield np.array(sigma1_list), np.array(sigma2_list), np.array(mu_list)


# 训练数据动态生成
def data_generator(m, X_train, encodings, obj_train, batch_size=64, num_pairs_per_problem=60):
    num_problems = X_train.shape[0]
    num_strategies = len(encodings)

    while True:  # 使得生成器可以无限产生数据
        #随机打乱
        problem_indices = np.random.permutation(num_problems)

        # 每次处理足够多的问题以填满至少一个batch
        for i in range(0, num_problems, batch_size):
            batch_problems = problem_indices[i:i + batch_size]
            sigma1_list = []
            sigma2_list = []
            mu_list = []
            for idx in batch_problems:
                if idx >= num_problems:
                    break
                problem = X_train.iloc[idx]
                p = pandas2array(problem)
                s_res = m.result[idx] if m.result is not None else np.array([None] * num_strategies)

                # 随机选择一部分策略对
                # selected_pairs = np.random.choice(num_strategies, (num_pairs_per_problem, 2), replace=False)
                # cn2
                selected_pairs = list(combinations(np.array(range(num_strategies)), 2))

                for j, k in selected_pairs:
                    # sigma1 = (p, strategy2array(encodings[j]))
                    s = strategy2array(encodings[j])[np.newaxis, :]
                    sigma1 = np.concatenate([p, s], axis=1)
                    # sigma2 = (p, strategy2array(encodings[k]))
                    s = strategy2array(encodings[k])[np.newaxis, :]
                    sigma2 = np.concatenate([p, s], axis=1)
                    result = compare_solutions(m, problem, encodings[j], encodings[k], obj_train[idx],
                                               j, k, s_res)  # 有问题！！！！！！！！！

                    if result == '1':
                        mu = [1, 0]
                    elif result == '2':
                        mu = [0, 1]
                    elif result == 'tie':
                        mu = [0.5, 0.5]
                    else:
                        continue  # 忽略不可比较的情况
                    sigma1_list.append(sigma1)
                    sigma2_list.append(sigma2)
                    mu_list.append(mu)

            if len(sigma1_list) > 0:
                yield sigma1_list, sigma2_list, mu_list


# 网络设置
class RM_MLP(nn.Module):
    def __init__(self, input_size):
        super(RM_MLP, self).__init__()
        self.relu = nn.ReLU()

        # 定义全连接层
        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.bn1 = nn.BatchNorm1d(input_size * 2)  # 批归一化层

        self.fc2 = nn.Linear(input_size * 2, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)  # 批归一化层

        self.fc3 = nn.Linear(input_size, 128)
        self.bn3 = nn.BatchNorm1d(128)  # 批归一化层

        self.fc4 = nn.Linear(128, 1)  # 输出层不通常不加批归一化

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 应用批归一化
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)  # 应用批归一化
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)  # 应用批归一化
        x = self.relu(x)

        x = self.fc4(x)
        x = torch.squeeze(x, -1)  # 去除最后一个维度如果它是1
        return x


# loss函数
class PreferenceBasedLoss(torch.nn.Module):
    def __init__(self):
        super(PreferenceBasedLoss, self).__init__()

    def forward(self, outputs1, outputs2, mu):
        # 将outputs转换为概率
        exp1 = torch.exp(outputs1)
        exp2 = torch.exp(outputs2)

        # 增加数值稳定性处理
        # max_output = torch.max(outputs1, outputs2)
        # exp1 = torch.exp(outputs1 - max_output)
        # exp2 = torch.exp(outputs2 - max_output)

        probabilities1 = exp1 / (exp1 + exp2)
        probabilities2 = exp2 / (exp1 + exp2)

        epsilon = 1e-8

        loss = -torch.mean(mu[:, 0] * torch.log(probabilities1 + epsilon) +
                           mu[:, 1] * torch.log(probabilities2 + epsilon))
        return loss


def training(use_current_model=False):
    # 定义模型，是否使用已存在的模型
    model = RM_MLP(input_size=input_size)
    if use_current_model and os.path.isfile(use_current_model):
        model.load_state_dict(torch.load(model_weight_name))
    else:
        pass

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    loss_fn = PreferenceBasedLoss()
    min_loss = float('inf')

    # 加载已有的损失历史或初始化新的列表
    if os.path.exists(loss_history_file):
        with open(loss_history_file, 'r') as file:
            try:
                loss_history = json.load(file)
            except json.JSONDecodeError:
                loss_history = []
    else:
        loss_history = []

    for i in tqdm(range(batch_num), desc='traning'):
        try:
            sigma1_list, sigma2_list, mu_list = next(gen)  # 从生成器获取数据
        except StopIteration:
            break  # 如果生成器没有更多的数据，结束循环

        # 数据转换为张量
        sigma1_tensor = torch.tensor(sigma1_list, dtype=torch.float32)
        sigma2_tensor = torch.tensor(sigma2_list, dtype=torch.float32)
        mu_tensor = torch.tensor(mu_list, dtype=torch.float32)

        sigma1_tensor = sigma1_tensor.squeeze(1)  # 去除中间维度
        sigma2_tensor = sigma2_tensor.squeeze(1)  # 去除中间维度
        loss_f = torch.nn.CrossEntropyLoss()

        # 计算损失
        outputs1 = model(sigma1_tensor)
        outputs2 = model(sigma2_tensor)
        loss = loss_fn(outputs1, outputs2, mu_tensor)
        # output = model(sigma1_tensor, sigma2_tensor)
        # loss = loss_f(output, mu_tensor.long())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 对梯度进行裁剪
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())
        print(f'Iteration {i + 1}, Loss: {loss.item()}')

        # 检查并保存最佳模型
        if (i + 1) % save_time == 0:
            print(f'Checking best model at iteration {i + 1}')
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), model_weight_name_best)
                print(f'New best model saved with loss {min_loss}')

    # 保存整个模型
    torch.save(model.state_dict(), model_weight_name)
    with open(loss_history_file, 'w') as f:
        json.dump(loss_history, f)


def draw():
    # 读取损失历史
    with open(loss_history_file, 'r') as file:
        loss_history = json.load(file)
    import matplotlib.pyplot as plt
    # 绘制损失图像
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def testing():
    topk_list = [1, 10, 20, 30]
    accuracy_dict = {}
    model = RM_MLP(input_size=input_size)
    # model.load_state_dict(torch.load(model_weight_name_best))
    model.load_state_dict(torch.load(model_weight_name_best))
    model.eval()
    strategys = m_mlopt.encoding
    m_mlopt.load_training_data("/home/ljj/project/mlopt/results/control/RLHF" +'/control_%d_' % 10 + 'test_data.pkl')
    # m_mlopt.load_training_data(EXAMPLE_NAME + 'test_data.pkl')
    test_problems = m_mlopt.X_train
    test_obj_train = m_mlopt.obj_train
    m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
    m_mlopt.cache_factors()
    correct_list = []

    time_topk_list = []
    time_topk_predict_list = []
    time_topk_solve_list = []
    for topk in topk_list:
        print("n_best:", topk)
        correct_list = []
        time_list = []
        time_predict_list = []
        time_solve_list = []

        for problem_idx in tqdm(range(test_problems.shape[0]), desc='test'):
            problem = test_problems.iloc[problem_idx]
            p = pandas2array(problem)
            rewards = []

            # 收集所有输入
            inputs = []
            # fix_inputs=[]
            #
            # s0=strategy2array(strategys[0])[np.newaxis, :]
            # fix_input = np.concatenate([p, s0], axis=1)

            for j in range(len(strategys)):
                s = strategy2array(strategys[j])[np.newaxis, :]  # 将策略转换为数组并增加一个新轴
                combined_input = np.concatenate([p, s], axis=1)  # 将问题和策略合并
                # fix_inputs.append(fix_input)
                inputs.append(combined_input)

            # 将列表转换为张量
            input_tensor = torch.tensor(np.concatenate(inputs, axis=0), dtype=torch.float32)  # 合并所有输入并转换为张量
            # fix_inputs = torch.tensor(np.concatenate(inputs, axis=0), dtype=torch.float32)
            # 一次性通过模型获取所有输出
            pre_time = time.time()
            outputs = model(input_tensor)
            pre_time = time.time() - pre_time

            rewards = outputs.detach().numpy().flatten().tolist()  # 转换输出为列表
            # 获取 topk 个最高奖励的策略的索引
            top_rewards_indices = torch.topk(torch.tensor(rewards), topk).indices

            # top_rewards_indices=(torch.argmax(outputs, dim=1) == 1).nonzero(as_tuple=True)[0]
            # if len(top_rewards_indices) == 0:
            #     top_rewards_indices = (torch.argmax(outputs, dim=1) == 2).nonzero(as_tuple=True)[0]

            m_mlopt._problem.populate(problem)

            results = []
            infeasibilities = []
            costs = []

            sol_time = 0
            # 求解所有策略并收集数据
            for k in list(top_rewards_indices):
            # for k in range(len(strategys)):
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

            # 判断次优性和不可行性是否满足约束
            if best_infeasibility <= stg.INFEAS_TOL and np.abs(best_suboptimality) <= stg.SUBOPT_TOL:
                idx_correct = 1
            else:
                idx_correct = 0

            correct_list.append(idx_correct)
            time_list.append(sol_time + pre_time)
            time_solve_list.append(sol_time)
            time_predict_list.append(pre_time)

        # 计算正确率
        accuracy = sum(correct_list) / len(correct_list)
        accuracy_dict[topk] = accuracy
        print(f"Accuracy: {accuracy:.2%}", " topk=", topk)

        # 时间
        print("time:", sum(time_list) / len(time_list))
        print("pre_time:", sum(time_predict_list) / len(time_predict_list))
        print("sol_time:", sum(time_solve_list) / len(time_solve_list))

        time_topk_list.append(sum(time_list) / len(time_list))
        time_topk_predict_list.append(sum(time_predict_list) / len(time_predict_list))
        time_topk_solve_list.append(sum(time_solve_list) / len(time_solve_list))

    # 保存数据到JSON文件
    with open(EXAMPLE_NAME + 'accuracy_results.json', 'w') as f:
        json.dump(accuracy_dict, f)
    with open(EXAMPLE_NAME + 'time_results.json', 'w') as f:
        json.dump(time_topk_list, f)
    with open(EXAMPLE_NAME + 'time_predict_results.json', 'w') as f:
        json.dump(time_topk_predict_list, f)
    with open(EXAMPLE_NAME + 'time_solve_results.json', 'w') as f:
        json.dump(time_topk_solve_list, f)


if __name__ == '__main__':
    start = time.time()
    desc = 'Online Control Example Testing'
    seed_test = 3
    tau = 1.0
    T = 10
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--horizon', type=int, default=T, metavar='N',
                        help='horizon length (default: 50)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    print("Horizon %d" % T_horizon)
    STORAGE_DIR = "/home/ljj/project/mlopt/results/control/setcover"
    mlopt_model_DIR = "/home/ljj/project/mlopt/results/control/setcover"
    EXAMPLE_NAME = STORAGE_DIR + '/control_%d_setcover' % T_horizon
    problem, cost_function_data = u.control_problem(T_horizon, tau=tau)

    # Load model
    n_best = 20
    m_mlopt = mlopt.Optimizer.from_file(mlopt_model_DIR + '/control_%d_setcover' % T_horizon + 'model')
    m_mlopt._learner.options['n_best'] = n_best

    m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
    p = m_mlopt.X_train.iloc[0]
    input_size = problem2array(m_mlopt.X_train.iloc[0]).shape[0] + strategy2array(m_mlopt.encoding[0]).shape[0]

    # 训练集生成器
    # gen = new_data_generator(m_mlopt, m_mlopt.X_train, m_mlopt.encoding, m_mlopt.obj_train, num_problems_per_call=32)

    # 这边调整训练batch的大小
    batch_size = 32  # batch_size的意思是：一个batch用多少个问题
    num_pairs_per_problem = 11  # 一个问题抽多少个策略对
    gen = data_generator(m_mlopt, m_mlopt.X_train, m_mlopt.encoding, m_mlopt.obj_train, batch_size=batch_size,
                         num_pairs_per_problem=num_pairs_per_problem)
    from tqdm import tqdm

    # 训练控制变量
    batch_num = math.floor(m_mlopt.X_train.shape[0] / batch_size ) # 定义总的训练次数
    save_time = 10  # 检查是否需要保存模型的次数

    total_trainning_pairs = batch_size * num_pairs_per_problem * batch_num
    total_pairs = m_mlopt.X_train.shape[0] * len(m_mlopt.encoding) * (len(m_mlopt.encoding) - 1) * 0.5
    print("traning pairs:", total_trainning_pairs)
    print("total pairs:", total_pairs)
    print("data rate:", total_trainning_pairs / total_pairs * 100, "%")

    model_weight_name = EXAMPLE_NAME + 'model_weight.pth'
    model_weight_name_best = EXAMPLE_NAME + 'best_model_weight.pth'

    # gpu
    # 检查CUDA是否可用，然后选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history_file = EXAMPLE_NAME + 'loss_history.json'  # 损失历史文件路径

    # 训练
    training(use_current_model=False)
    # 画图
    draw()
    # 测试模型性能
    testing()

    print("用时", time.time() - start)
