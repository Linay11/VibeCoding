# Needed for slurm
import os
import sys
import time
import uuid
from typing import List

import math
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import LambdaLR

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
import pdb
warnings.filterwarnings("ignore")
seed = int(time.time()) + os.getpid()
np.random.seed(seed)


def lr_lambda(epoch):
    return 0.9 ** ((epoch + 1) // 2)


def f(x):
    epsilon = 1e-8
    return (- x * np.log(1 - x + epsilon) + x) * 15 + 1


def norm(data):
    # data = [d * 1e2 for d in data]
    # softmax_data = np.exp(data) / np.sum(np.exp(data))
    # log_softmax_data = np.log(softmax_data)
    log_data = -np.log(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_log_data = scaler.fit_transform(log_data).flatten()
    return normalized_log_data


def cal_correct_infeas_subopt(infea, costs, target_cost):
    feasible_indices = np.where(infea <= stg.INFEAS_TOL)[0]
    if len(feasible_indices) > 0:
        # 选择成本最优的策略
        if m_mlopt._problem.sense() == Minimize:
            best_idx = feasible_indices[np.argmin(costs[feasible_indices])]
        elif m_mlopt._problem.sense() == Maximize:
            best_idx = feasible_indices[np.argmax(costs[feasible_indices])]
        else:
            raise ValueError('Objective type not understood')
    else:
        # 所有策略都不可行，选择不可行性最小的
        best_idx = np.argmin(infea)

    best_infeasibility = infea[best_idx]
    best_cost = costs[best_idx]
    best_suboptimality = utils.suboptimality(best_cost, target_cost, m_mlopt._problem.sense())
    if best_suboptimality < 0:
        best_suboptimality = 0

    # 判断次优性和不可行性是否满足约束
    if best_infeasibility <= stg.INFEAS_TOL and np.abs(best_suboptimality) <= stg.SUBOPT_TOL:
        idx_correct = 1
    else:
        idx_correct = 0

    return idx_correct, best_infeasibility, best_suboptimality


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


def get_score(res, suboptim, w=1):
    if suboptim <= 0:
        suboptim = 0
    score = np.abs(suboptim) + res['infeasibility'] * w
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


def cal_score(m, target_cost, res, j):
    correct = 0
    s = m.encoding
    if res is None:
        res = m._problem.solve(strategy=s[j])
    else:
        res = res[j]

    subopt = utils.suboptimality(res['cost'], target_cost, m._problem.sense())
    if subopt <= 0:
        subopt = 0
    score = get_score(res, subopt)
    if res['infeasibility'] <= stg.INFEAS_TOL and subopt <= stg.SUBOPT_TOL:
        correct = 1
    return score, correct


def cn2_data_generator(m, X_train, encodings, obj_train, results=None, num_problems_per_call=10, num_pairs_per_problem=10):
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
        delta_total_list = []

        # 对于每一个问题
        for idx in current_batch_indices:
            # 获取问题的基本信息
            problem = X_train.iloc[idx]
            p = pandas2array(problem)
            target_cost = obj_train[idx]
            res = results[idx]

            # 随机挑选出策略
            if num_strategies < 2:
                raise ValueError("策略数量过少，无法形成策略对。")
            num_s_to_select = min(num_strategies // 2, num_pairs_per_problem)  # 选择最小值以避免选择过多的策略
            selected_pairs = np.random.choice(num_strategies, num_s_to_select * 2, replace=False)

            # 计算所选策略的分数
            scores_raw = []
            for s_idx in selected_pairs:
                info = cal_score(m, target_cost, res, s_idx)
                scores_raw.append(info[0])
            scores = np.array([1e-5 if x <= 1e-5 else x for x in scores_raw])
            scores = norm(np.array(scores)) # 从这里开始分数越高越好

            sigma1_list = []
            sigma2_list = []
            mu_list = []
            delta_list = []

            # 生成顺序策略对
            for j in range(0, len(selected_pairs) - 1, 2):
                s1_idx = selected_pairs[j]
                s2_idx = selected_pairs[j + 1]

                s1 = strategy2array(encodings[s1_idx])[np.newaxis, :]
                s2 = strategy2array(encodings[s2_idx])[np.newaxis, :]
                sigma1 = np.concatenate([p, s1], axis=1)
                sigma2 = np.concatenate([p, s2], axis=1)

                score1 = scores[j]
                score2 = scores[j+1]

                delta = (score1 - score2)

                # 根据分数差距决定mu
                if np.abs(score1 - score2) <= stg.INFEAS_TOL * 0.1:
                    mu = [0.5, 0.5]
                    delta = 0
                elif score1 > score2:
                    mu = [1, 0]
                    delta = f(delta)  # f可能要改，因为score1-score2为1会出问题
                else:
                    mu = [0, 1]
                    delta = f(delta)

                if np.isnan(delta):
                    pdb.set_trpace()  # 如果delta是nan，设置断点

                if np.isinf(delta):
                    pdb.set_trpace()  # 如果delta是inf，设置断点


                sigma1_list.append(sigma1)
                sigma2_list.append(sigma2)
                mu_list.append(mu)
                delta_list.append(delta)

            sigma1_total_list.append(sigma1_list)
            sigma2_total_list.append(sigma2_list)
            mu_total_list.append(mu_list)
            delta_total_list.append(delta_list)

        if sigma1_total_list:
            yield np.array(sigma1_total_list), np.array(sigma2_total_list), np.array(mu_total_list), np.array(
                delta_total_list)

# def seq_data_generator(m, X_train, encodings, obj_train, results=None, num_problems_per_call=10):
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
#         sigma1_total_list = []
#         sigma2_total_list = []
#         mu_total_list = []
#         delta_total_list = []
#         satisfy_total_indices = []
#         sorted_total_indices = []
#
#         for idx in current_batch_indices:
#             problem = X_train.iloc[idx]
#             p = pandas2array(problem)
#             target_cost = obj_train[idx]
#             res = results[idx]
#
#             information = np.array([cal_score(m, target_cost, res, j) for j in range(num_strategies)])
#             satisfy_indices = np.where(information[:, 1] == 1)[0]  #满足条件的策略的索引
#             satisfy_total_indices.append(satisfy_indices)
#
#             scores_raw = information[:, 0]
#             sorted_indices = np.argsort(scores_raw)  # 分数从低到高排序的索引
#             sorted_total_indices.append(sorted_indices)
#             scores = np.array([1e-5 if x <= 1e-5 else x for x in scores_raw])
#             scores = norm(scores)  # 从这里开始分数越高越好
#
#             sigma1_list = []
#             sigma2_list = []
#             mu_list = []
#             delta_list = []
#
#             # 生成顺序策略对
#             for j in range(len(sorted_indices) - 1):
#                 s1_idx = sorted_indices[j]
#                 s2_idx = sorted_indices[j + 1]
#
#                 s1 = strategy2array(encodings[j])[np.newaxis, :]
#                 s2 = strategy2array(encodings[j + 1])[np.newaxis, :]
#                 sigma1 = np.concatenate([p, s1], axis=1)
#                 sigma2 = np.concatenate([p, s2], axis=1)
#
#                 score1 = scores[s1_idx]
#                 score2 = scores[s2_idx]
#
#                 delta = (score1 - score2)
#
#                 # 根据分数差距决定mu
#                 if np.abs(scores_raw[s1_idx] - scores_raw[s2_idx]) <= stg.INFEAS_TOL * 0.1:
#                     mu = [0.5, 0.5]
#                     delta = 0
#                 elif score1>score2:
#                     mu = [1, 0]
#                     delta = f(delta)
#                 else:
#                     mu = [0, 1]
#                     delta = f(delta)
#
#                 sigma1_list.append(sigma1)
#                 sigma2_list.append(sigma2)
#                 mu_list.append(mu)
#                 delta_list.append(delta)
#                 if j + 1 == len(sorted_indices) - 1:
#                     sigma1_list.append(sigma2)
#
#             sigma1_total_list.append(sigma1_list)
#             sigma2_total_list.append(sigma2_list)
#             mu_total_list.append(mu_list)
#             delta_total_list.append(delta_list)
#
#         # # 打乱所有列表
#         # combined = list(zip(sigma1_total_list, sigma2_total_list, mu_total_list, delta_total_list, satisfy_total_indices, sorted_total_indices))
#         # np.random.shuffle(combined)
#         # sigma1_total_list, sigma2_total_list, mu_total_list, delta_total_list, satisfy_total_indices, sorted_total_indices = zip(*combined)
#
#         if sigma1_total_list:
#             yield np.array(sigma1_total_list), np.array(sigma2_total_list), np.array(mu_total_list), np.array(
#                 delta_total_list), np.array(satisfy_total_indices), np.array(sorted_total_indices)


class RM_MLP(nn.Module):
    def __init__(self, input_size):
        super(RM_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        # 定义全连接层
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出层不通常不加批归一化

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.dropout(x))

        x = self.fc2(x)
        x = self.relu(self.dropout(x))

        x = self.fc3(x)
        return x


class MyTransformer(nn.Module):
    def __init__(self, input_size, nhead=3, dim_feedforward=512, dropout=0.1):
        super(MyTransformer, self).__init__()
        self.input_size = input_size
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, dropout=dropout,
                                                       activation='gelu',batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=2)

        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x


# loss函数
class PreferenceBasedLoss(torch.nn.Module):
    def __init__(self, size, num_s, w=0.98):
        super(PreferenceBasedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.size = size
        self.num_s = num_s
        self.w = w

    def forward(self, outputs1, outputs2, mu, delta):
        outputs1 = outputs1.squeeze(2)
        outputs2 = outputs2.squeeze(2)

        exp1 = torch.exp(outputs1)
        exp2 = torch.exp(outputs2)

        probabilities1 = exp1 / (exp1 + exp2)
        probabilities2 = exp2 / (exp1 + exp2)

        epsilon = 1e-8

        mu1 = mu[:, :, 0]
        mu2 = mu[:, :, 1]
        loss = -torch.mean(mu1 * torch.log(probabilities1 + epsilon) +
                           mu2 * torch.log(probabilities2 + epsilon))
        diff = outputs1 - outputs2
        delta = delta.squeeze(2)
        mse_loss = self.mse_loss(diff, delta)

        total_loss = loss * self.w + mse_loss * (1 - self.w)


        return total_loss


def training(model, optimizer, scheduler, num_s, batch_size, min_loss, old_outputs=None):
    total_loss = 0
    loss_fn = PreferenceBasedLoss(batch_size, num_s, mse_w)

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
            sigma1_list, sigma2_list, mu_list, delta_list = next(train_gen)  # 从生成器获取数据
        except StopIteration:
            break  # 如果生成器没有更多的数据，结束循环

        # 数据转换为张量
        optimizer.zero_grad()
        sigma1_tensor = torch.tensor(sigma1_list, dtype=torch.float32)
        sigma2_tensor = torch.tensor(sigma2_list, dtype=torch.float32)
        mu_tensor = torch.tensor(mu_list, dtype=torch.float32)
        delta_tensor = torch.tensor(delta_list, dtype=torch.float32)

        delta_tensor = delta_tensor.unsqueeze(2)
        sigma1_tensor = sigma1_tensor.squeeze(2)
        sigma2_tensor = sigma2_tensor.squeeze(2)

        # 计算损失
        outputs1 = model(sigma1_tensor)
        outputs2 = model(sigma2_tensor)

        loss = loss_fn(outputs1, outputs2, mu_tensor, delta_tensor)

        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())
        total_loss += loss.item()
        # print(f'Iteration {i + 1}, Loss: {loss.item()}')

        # 检查并保存最佳模型
        if (i + 1) % save_time == 0:
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), model_weight_name_best)
                print(f'New best model saved with loss {min_loss}')

    scheduler.step()
    print('average loss: ', total_loss / batch_num)
    # 保存整个模型
    torch.save(model.state_dict(), model_weight_name)
    with open(loss_history_file, 'w') as f:
        json.dump(loss_history, f)

    return old_outputs, min_loss


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


def testing(model, test_problems=None, test_obj_train=None):
    topk_list = [1, 3, 5] # 按实验要求改
    accuracy_list = []
    time_list = []
    infeas_list = []
    subopt_list = []
    # model.load_state_dict(torch.load(model_weight_name_best))
    model.load_state_dict(torch.load(model_weight_name))
    model.eval()
    strategys = m_mlopt.encoding
    if test_problems is None or test_obj_train is None:
        m_mlopt.load_training_data(
            "." + '/control_%d_setcover_' % T_horizon + 'test_data.pkl')
        # m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
        test_problems = m_mlopt.X_train
        test_obj_train = m_mlopt.obj_train
    m_mlopt.load_training_data(EXAMPLE_NAME + 'data_filtered.pkl')
    m_mlopt.cache_factors()

    # results_test = m_mlopt._problem.solve_parametric(test_problems, parallel=True)

    for topk in topk_list:
        print("n_best:", topk)
        correct_list = []
        batch_test_num = math.floor(test_problems.shape[0] / batch_size)
        sol_time = 0
        pred_time = 0

        topk_infeas_list = []
        topk_subopt_list = []
        for batch in tqdm(range(batch_test_num)):
            inputs_total = []

            # 把一个batch_size的问题的输出搞出来
            for idx in range(batch_size):
                inputs = []
                problem_idx = batch * batch_size + idx
                problem = test_problems.iloc[problem_idx]
                p = pandas2array(problem)
                for j in range(len(strategys)):
                    s = strategy2array(strategys[j])[np.newaxis, :]  # 将策略转换为数组并增加一个新轴
                    combined_input = np.concatenate([p, s], axis=1)  # 将问题和策略合并
                    inputs.append(combined_input)
                inputs_total.append(inputs)

            # 过网络，得到前topk个策略
            input_tensor = torch.tensor(inputs_total, dtype=torch.float32).squeeze(2)
            temp = time.time()
            outputs = model(input_tensor)
            pred_time += time.time() - temp
            rewards = outputs.detach().squeeze(2)
            top_rewards_indices = torch.topk(torch.tensor(rewards), k=topk, dim=1).indices

            # 对于batch_size里的问题
            for idx in range(batch_size): # 每个问题
                problem_idx = batch * batch_size + idx
                problem = test_problems.iloc[problem_idx]
                m_mlopt._problem.populate(problem)
                results = []
                infeasibilities = []
                costs = []

                # k个策略
                for k in list(top_rewards_indices[idx]):
                    s = strategys[k]
                    res = m_mlopt._problem.solve(strategy=s)
                    sol_time += res['time']
                    results.append(res)
                    infeasibilities.append(res['infeasibility'])
                    costs.append(res['cost'])

                infeasibilities = np.array(infeasibilities)
                costs = np.array(costs)
                idx_correct, infeas ,subopt = cal_correct_infeas_subopt(infeasibilities, costs, target_cost=test_obj_train[problem_idx])
                correct_list.append(idx_correct)
                topk_infeas_list.append(infeas)
                topk_subopt_list.append(subopt)

        accuracy = sum(correct_list) / len(correct_list)
        accuracy_list.append(accuracy)

        avg_time = (pred_time+sol_time)/len(correct_list)
        time_list.append(avg_time)

        avg_infeas = np.mean(topk_infeas_list)
        infeas_list.append(avg_infeas)

        topk_subopt_list = np.array(topk_subopt_list)
        subopt_real = topk_subopt_list[np.where(np.array(topk_infeas_list) <= stg.INFEAS_TOL)[0]]
        if any(subopt_real):
            avg_subopt = np.mean(subopt_real)
        else:
            avg_subopt = np.nan
        subopt_list.append(avg_subopt)

        print(f"Accuracy: {accuracy:.2%}", " topk=", topk)

    df = pd.DataFrame({
        'topk': topk_list,
        'acc': accuracy_list,
        'infeas': infeas_list,
        'subopt': subopt_list,
        'time': time_list
    })
    df.to_csv(EXAMPLE_NAME+"RLHF_seq_result.csv", index=False)

def valid(model):
    """
    嫌麻烦，就不计算acc了
    :param model:
    :return:
    """
    model.eval()
    loss_fn = PreferenceBasedLoss(batch_size, num_s, mse_w)
    loss_total = torch.tensor(0, dtype=torch.float32)
    for i in tqdm(range(batch_valid_num), desc='valid'):
        try:
            sigma1_list, sigma2_list, mu_list, delta_list = next(valid_gen)  # 从生成器获取数据
        except StopIteration:
            break  # 如果生成器没有更多的数据，结束循环

        optimizer.zero_grad()
        sigma1_tensor = torch.tensor(sigma1_list, dtype=torch.float32)
        sigma2_tensor = torch.tensor(sigma2_list, dtype=torch.float32)
        mu_tensor = torch.tensor(mu_list, dtype=torch.float32)
        delta_tensor = torch.tensor(delta_list, dtype=torch.float32)

        delta_tensor = delta_tensor.unsqueeze(2)
        sigma1_tensor = sigma1_tensor.squeeze(2)
        sigma2_tensor = sigma2_tensor.squeeze(2)

        # 计算损失
        outputs1 = model(sigma1_tensor)
        outputs2 = model(sigma2_tensor)

        loss = loss_fn(outputs1, outputs2, mu_tensor, delta_tensor)

        loss_total += loss

    avg_loss = loss_total / batch_valid_num
    print(f"Valid Avg loss: {avg_loss:.4f}")


if __name__ == '__main__':
    start = time.time()
    desc = 'Online Control Example Testing'
    #seed_test = 3
    tau = 1.0
    T = 40
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--horizon', type=int, default=T, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    print("Horizon %d" % T_horizon)
    STORAGE_DIR = "/home/ljj/project/mlopt/results/control/setcover"
    mlopt_model_DIR = "/home/ljj/project/mlopt/results/control/setcover"
    EXAMPLE_NAME = STORAGE_DIR + '/control_%d_setcover_ntraj_10000' % T_horizon
    # EXAMPLE_NAME = STORAGE_DIR + '/control_%d_' % T_horizon
    # problem, cost_function_data = u.control_problem(T_horizon, tau=tau)

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
    num_s = len(m_mlopt.encoding)
    batch_size = 128  # batch_size的意思是：一个batch用多少个问题

    train_num = round(0.8 * len(m_mlopt.X_train))
    valid_num = round(0.1 * len(m_mlopt.X_train))

    X_train = m_mlopt.X_train[:train_num]
    y_train = m_mlopt.obj_train[:train_num]
    X_res_train = m_mlopt.result[:train_num] if m_mlopt.result is not None else None

    X_valid = m_mlopt.X_train[train_num:train_num + valid_num]
    y_valid = m_mlopt.obj_train[train_num:train_num + valid_num]
    X_res_valid = m_mlopt.result[train_num:train_num + valid_num] if m_mlopt.result is not None else None

    # X_test = m_mlopt.X_train[train_num + valid_num:]
    # y_test = m_mlopt.obj_train[train_num + valid_num:]
    # X_res_test = m_mlopt.result[train_num + valid_num:] if m_mlopt.result is not None else None

    train_gen = cn2_data_generator(m_mlopt, X_train, m_mlopt.encoding, y_train, X_res_train, batch_size,num_pairs_per_problem=10)
    valid_gen = cn2_data_generator(m_mlopt, X_valid, m_mlopt.encoding, y_valid, X_res_valid, batch_size, num_pairs_per_problem=10)
    from tqdm import tqdm

    # 训练控制变量
    batch_num = math.floor(X_train.shape[0] / batch_size)
    batch_valid_num = math.floor(X_valid.shape[0] / batch_size)
    save_time = 10  # 检查是否需要保存模型的次数
    epoch = 10
    model = MyTransformer(input_size=input_size)
    # model = RM_MLP(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = LambdaLR(optimizer, lr_lambda)

    mse_w = 1
    model_dir = '.' + '/control_%d_' % T_horizon
    model_weight_name = model_dir + f'model_weight_{mse_w}.pth'
    model_weight_name_best = model_dir + f'best_model_weight_{mse_w}.pth'

    # gpu
    # 检查CUDA是否可用，然后选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_history_file = EXAMPLE_NAME + 'loss_history.json'  # 损失历史文件路径
    old_outputs = None

    # 训练
    min_loss = float('inf')
    use_current_model = False
    if use_current_model and os.path.isfile(model_weight_name_best):
        model.load_state_dict(torch.load(model_weight_name_best))
    else:
        pass

    # 训练
    for i in range(epoch):
        print("Epoch %d" % i)
        old_outputs, min_loss = training(model, optimizer, scheduler, num_s, batch_size, min_loss, old_outputs)
        valid(model)
    # draw()

    # 测试模型性能
    testing(model)

    print("用时", time.time() - start)
