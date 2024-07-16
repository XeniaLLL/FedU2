import os.path
from typing import Union, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from scipy.sparse import csr_matrix


class Chapman(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: str, label: str = 'Rhythm',
                 attributes: Union[str, Sequence[str]] = ('Rhythm', 'PatientAge', 'Gender'),
                 ml: int = 0, num_constrains: int = 0, q: float = 0., alpha: float = 0.):
        self.root = root
        self.ml = ml
        self.q = q
        self.alpha = alpha
        self.num_constrains = num_constrains
        self.attributes = attributes if isinstance(attributes, str) else list(attributes)
        diagnostics = pd.read_excel(
            os.path.join(root, 'Diagnostics.xlsx'),
            dtype={
                'FileName': str,
                'Rhythm': 'category',
                'PatientAge': int,
                'Gender': 'category'
            }
        )
        self.rhythm_names: list[str] = diagnostics['Rhythm'].cat.categories.to_list()
        self.gender_names: list[str] = diagnostics['Gender'].cat.categories.to_list()
        self.filenames: list[str] = diagnostics['FileName'].to_list()
        diagnostics['Rhythm'] = diagnostics['Rhythm'].cat.codes
        diagnostics['Gender'] = diagnostics['Gender'].cat.codes
        self.targets = torch.tensor(
            diagnostics[label].to_numpy(),
            dtype=torch.int64
        )  # shape(N,) or shape(N, attr_num)
        if os.path.exists(os.path.join(root, 'ECGData.pth')):
            self.data = torch.load(os.path.join(root, 'ECGData.pth'))
        else:
            data = []
            for filename in tqdm(self.filenames, desc='Reading Chapman', leave=False):
                ecg_df = pd.read_csv(
                    os.path.join(root, 'ECGData', f'{filename}.csv'),
                    dtype=np.float32,
                    nrows=5000
                )
                # shape(sequence_length, channel)
                data.append(torch.tensor(ecg_df.to_numpy(), dtype=torch.float32))
            self.data = torch.stack(data, dim=0)  # shape(N, sequence_length, channel)

        self.W = self.get_W()

    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number
        # Return
            transtive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)
        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

    def generate_random_pair(self, y, num, q):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, self.l - 1)
            tmp2 = random.randint(0, self.l - 1)
            ii = np.random.uniform(0, 1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                if ii >= q:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
            else:
                if ii >= q:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
                else:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
            num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_ml(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_cl(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] != y[tmp2]:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def get_W(self):
        '''
        constrained clustering
        num_constrains: 约束多少个sample pair
        Y: label
        ml: link 方式,约束must link (1), random w/o prior (0), and cannot link (-1)
        q: random ratio

        return
        ml_ind1:
        ml_ind2:
        cl_ind1:
        cl_ind2:
        '''
        if self.ml == 0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair(self.targets.numpy().tolist(),
                                                                           self.num_constrains, self.q)
            if self.q == 0:
                ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2,
                                                                             self.data.shape[0])
        elif self.ml == 1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_ml(self.targets.numpy().tolist(),
                                                                              self.num_constrains)
        elif self.ml == -1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_cl(self.targets.numpy().tolist(),
                                                                              self.num_constrains)
        else:
            raise ValueError(f"{self.ml} is not right")
        print("\nNumber of ml constraints: %d, cl constraints: %d.\n " % (len(ml_ind1), len(cl_ind1)))

        # W = np.zeros([len(self.X), len(self.X)])
        # for i in range(len(ml_ind1)):
        #    W[ml_ind1[i], ml_ind2[i]] = 1
        #    W[ml_ind2[i], ml_ind1[i]] = 1
        # for i in range(len(cl_ind1)):
        #    W[cl_ind1[i], cl_ind2[i]] = -1
        #    W[cl_ind2[i], cl_ind1[i]] = -1
        # W = csr_matrix(W)

        # if self.num_constrains > 0:
        # if False:
        #     ml_ind1 = np.load("source/data1_pos.npy")
        #     ml_ind2 = np.load("source/data2_pos.npy")
        #     cl_ind1 = np.load("source/data1_neg.npy")
        #     cl_ind2 = np.load("source/data2_neg.npy")

        ind1 = np.concatenate([ml_ind1, ml_ind2, cl_ind1, cl_ind2])
        ind2 = np.concatenate([ml_ind2, ml_ind1, cl_ind2, cl_ind1])
        data = np.concatenate([np.ones(len(ml_ind1) * 2), np.ones(len(cl_ind1) * 2) * -1])
        W = csr_matrix((data, (ind1, ind2)), shape=(self.data.shape[0], self.data.shape[0]))
        W = W.tanh().rint()
        return W, ml_ind1, ml_ind2, cl_ind1, cl_ind2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index], torch.from_numpy(self.W[index][:,index] * self.alpha) # careful  W 的读取和格式转换

    def __len__(self) -> int:
        return self.data.shape[0]
