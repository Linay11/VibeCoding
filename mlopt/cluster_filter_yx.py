from joblib import Parallel, delayed
import mlopt.settings as stg
import numpy as np
import mlopt.utils as u
from tqdm.auto import tqdm
import pickle as pkl
import pandas as pd
import copy
import csv
from sklearn.cluster import KMeans

def read_adj_matrix_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.values

def strategy_degeneration(theta, obj_train, encoding, problem):
    """Compute best strategy between the ones in encoding."""

    problem.populate(theta)  # Populate parameters

    # Serial solution over the strategies
    results = [problem.solve(strategy=strategy) for strategy in encoding]



    # Compute cost degradation
    degradation = []
    for r in results:

        cost = r['cost']

        if r['infeasibility'] > stg.INFEAS_TOL:
            cost = np.inf

        diff = np.abs(cost - obj_train)
        if np.abs(obj_train) > stg.DIVISION_TOL:
            diff /= np.abs(obj_train)
        degradation.append(diff)


    return degradation


def vertex_cover_bipartite(adj_matrix):
    '''集合覆盖问题贪心求解，返回最有用的策略'''
    left_nodes, right_nodes = adj_matrix.shape
    uncovered = set(range(left_nodes))
    removed_right_nodes = set()
    label_dict = {}
    while uncovered:
        best_right_node = -1
        max_covered = 0
        for j in range(right_nodes):
            if j in removed_right_nodes:
                continue
            covered = {i for i in uncovered if adj_matrix[i][j] == 1}
            if len(covered) > max_covered:
                max_covered = len(covered)
                best_right_node = j

        if best_right_node == -1:
            break

        removed_right_nodes.add(best_right_node)
        cover_set = {i for i in uncovered if adj_matrix[i][best_right_node] == 1}
        uncovered -= cover_set
        label_dict[best_right_node] = cover_set
    removed_right_nodes=list(removed_right_nodes)

    return removed_right_nodes,label_dict




class Cluster_Filter(object):
    """Strategy filter."""

    def __init__(self,
                 X_train=None,
                 y_train=None,
                 obj_train=None,
                 encoding=None,
                 problem=None):
        """Initialize strategy condenser."""
        self.X_train = X_train
        self.y_train = y_train
        self.encoding = encoding
        self.obj_train = obj_train
        self.problem = problem


    def assign_samples_new(self, discarded_samples, selected_strategies,label_set,
                       batch_size, parallel=True):
        """
        Assign samples to strategies choosing the ones minimizing the cost.
        """

        # Backup strategies labels and encodings
        #  self.y_full = self.y_train

        # Reassign y_labels
        # selected_strategies: find index where new labels are
        # discarded_strategies: -1

        for key in label_set:
            for i in label_set[key]:
                self.y_train[i] = key
        selected_strategies_array = np.array(selected_strategies)
        new_y_train = []
        for label in self.y_train:
            if label in selected_strategies:
                indices = np.where(selected_strategies_array == label)[0]
                if indices.size > 0:
                    new_y_train.append(indices[0])
                else:
                    new_y_train.append(-1)  # 没有匹配项，添加-1
                    raise ValueError("出现-1，没有完全cover")
            else:
                new_y_train.append(-1)  # 标签不在选择的策略中
                raise ValueError("出现-1，没有完全cover")

        self.y_train = np.array(new_y_train)

        # Assign discarded samples and compute degradation
        degradation = np.zeros(len(discarded_samples))


        return degradation



    def select_strategies_setcover(self,degeneration_matrix_01):
        """基于setcover的选策略"""

        n_samples = len(self.X_train)
        n_strategies = len(self.encoding)

        stg.logger.info("Selecting strategies by solving setcover problem")
        #X_train是样本的问题参数，y_train是每个样本对应的策略的索引


        selected_strategies,label_set = vertex_cover_bipartite(degeneration_matrix_01)

        stg.logger.info("Selected %d strategies" % len(selected_strategies))

        return selected_strategies,label_set

    def degenaration_matrix_generation(self,batch_size, parallel=True):
        '''返回策略覆盖实例的邻接矩阵'''
        degeneration_matrix = [] #存储每个样本对应的策略的degeneration矩阵
        # for i in tqdm(range(len(self.X_train))):
        #     degeneration = strategy_degeneration(self.X_train.iloc[i], self.obj_train[i],
        #                                self.encoding, self.problem)
        #     degeneration_matrix.append(degeneration)

        n_jobs = u.get_n_processes() if parallel else 1
        #n_jobs = 8
        stg.logger.info("Assign samples to selected strategies (n_jobs = %d)"
                        % n_jobs)

        degeneration_matrix = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(strategy_degeneration)(self.X_train.iloc[i], self.obj_train[i],
                                   self.encoding, self.problem)
            for i in tqdm(range(len(self.X_train)))
        )

        degeneration_matrix = np.array(degeneration_matrix)

        degeneration_matrix_01 = np.where(degeneration_matrix < stg.INFEAS_TOL, 1, 0)
        return degeneration_matrix_01

    def filter(self,
               samples_fraction=stg.FILTER_STRATEGIES_SAMPLES_FRACTION,
               max_iter=stg.FILTER_MAX_ITER,
               batch_size=stg.JOBLIB_BATCH_SIZE,
               parallel=True):
        """Filter strategies."""

        # Backup strategies labels and encodings
        self.X_train_full = copy.deepcopy(self.X_train)
        self.y_full = copy.deepcopy(self.y_train)
        self.encoding_full = copy.deepcopy(self.encoding)

        n_samples = len(self.X_train)

        #获得所有样本与其适用的约简策略矩阵
        degeneration_matrix_01 = self.degenaration_matrix_generation(batch_size=batch_size, parallel=parallel)
        np.savetxt("T_40_10000_degra.csv", degeneration_matrix_01, delimiter=",", fmt="%d")
        # # 或者直接读取邻接矩阵
        # file_path = 'T_10_10000_degeneration.csv'
        # degeneration_matrix_01 = read_adj_matrix_from_csv(file_path)


        #去掉行内所有元素为0的样本（所有约简策略都不适用），很神奇，也就是说label算出来是某个策略，但这个策略用在它上面是不可行
        # 检查是否有全零行
        zero_rows = np.all(degeneration_matrix_01 == 0, axis=1)
        has_zero_row = np.any(zero_rows)
        # 根据是否有全零行执行不同的代码
        if has_zero_row:
            print("矩阵中有全零行。执行相关操作...")
            # 使用 numpy.any() 沿着列方向检查 (axis=1)，找出至少有一个非零元素的行
            non_zero_row_mask = np.any(degeneration_matrix_01 != 0, axis=1)
            # 选择这些行
            degeneration_matrix_01 = degeneration_matrix_01[non_zero_row_mask]

            # 使用 numpy.where() 获取满足条件的行索引
            non_zero_row_indices = np.where(non_zero_row_mask)[0]
            # 使用列表推导来过滤出对应非全零行的元素
            self.X_train = self.X_train[non_zero_row_mask]
            self.y_train = [self.y_train[i] for i in non_zero_row_indices]
            self.obj_train = [self.obj_train[i] for i in non_zero_row_indices]
            # 重新备份样本减少后的y
            self.y_full = copy.deepcopy(self.y_train)
            np.savetxt("T_10_10000_degeneration_filtered.csv", degeneration_matrix_01, delimiter=",", fmt="%d")
            n_samples = len(self.X_train)

        else:
            print("矩阵中没有全零行。不执行特定操作。")



        for k in range(max_iter):

            selected_strategies,label_set = self.select_strategies_setcover(degeneration_matrix_01)  #返回选择的策略，以及这些策略cover的实例组成的字典

            # Reassign encodings and labels
            self.encoding = [self.encoding[i] for i in selected_strategies]

            # Find discarded samples
            discarded_samples = np.array([i for i in range(n_samples)
                                          if self.y_train[i]
                                          not in selected_strategies])

            #stg.logger.info("Samples fraction at least %.3f %%" % (100 * samples_fraction))
            stg.logger.info("Discarded strategies for %d samples (%.2f %%)" %
                            (len(discarded_samples),
                             (100 * len(discarded_samples) / n_samples)))

            # Reassign discarded samples to selected strategies
            degradation = self.assign_samples_new(discarded_samples,
                                              selected_strategies,
                                                  label_set,
                                              batch_size=batch_size,
                                              parallel=parallel)


            if len(degradation) > 0:
                stg.logger.info("\nAverage cost degradation = %.2e %%" %
                                (100 * np.mean(degradation)))
                stg.logger.info("Max cost degradation = %.2e %%" %
                                (100 * np.max(degradation)))

                if np.mean(degradation) > stg.FILTER_SUBOPT:

                    stg.logger.info("Mean degradation too high")
                    self.y_train = self.y_full
                    self.encoding = self.encoding_full
                else:
                    stg.logger.info("Acceptable degradation found")
                    break
            else:
                stg.logger.info("No more discarded points.")
                break

        if k == max_iter - 1:
            self.y_train = self.y_full
            self.encoding = self.encoding_full
            stg.logger.warning("No feasible filtering found.")

        return self.y_train, self.encoding , self.X_train,self.obj_train
