import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import List
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from utils.mol import smiles2graph


def csv_to_ndarray(filepath):
    df = pd.read_csv(filepath, sep=',', header=None)

    return df.values


def adj_to_graph(adj: np.ndarray):
    edges_list = []
    nodes_feature = []
    n_nodes, _ = adj.shape
    for u in range(n_nodes):
        nodes_feature.append(adj[u])
        for v in range(n_nodes):
            weight = adj[u][v]
            if weight != 0:
                edges_list.append([u, v])
                edges_list.append([v, u])

    nodes_feature = torch.tensor(nodes_feature, dtype=torch.float)
    edges_index: torch.Tensor = torch.tensor(edges_list, dtype=torch.long)
    edges_index = edges_index.t().contiguous()

    graph = Data(x=nodes_feature, edge_index=edges_index)

    return graph


def create_graph_list(filename, device) -> List[Data]:
    """
    Convert each smiles of drug to graph.
    :return: list of Data object
    """
    data_list = []
    pf = pd.read_csv(filename, sep=',', header=None)
    data = pf.values
    for row in data[1:]:
        smiles = row[3]
        graph = smiles2graph(smiles, device)
        data_list.append(graph)

    return data_list


def adj_to_edge_list(filename):
    pf = pd.read_csv(filename, sep=',', header=None)
    adj = pf.values
    n_drugs, n_dis = adj.shape
    edge_list = list()
    y = []

    for drug_id in range(n_drugs):
        for dis_id in range(n_dis):
            weight = adj[drug_id][dis_id]
            if weight != 0:
                edge_list.append([drug_id, dis_id])
                y.append(1)
            else:
                edge_list.append([drug_id, dis_id])
                y.append(0)

    return np.array(edge_list), np.array(y)


def create_dataset(dataset_file, k_fold):
    edge_list, y = adj_to_edge_list(dataset_file)
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=47)
    print(skf)
    x_train_folds = list()
    x_test_folds = list()

    y_train_folds = list()
    y_test_folds = list()

    for train_idx, test_idx in skf.split(edge_list, y):
        print("Train:", len(train_idx), "Test", len(test_idx))
        x_train, x_test = edge_list[train_idx], edge_list[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        x_train_folds.append(x_train)
        y_train_folds.append(y_train)

        x_test_folds.append(x_test)
        y_test_folds.append(y_test)

    return x_train_folds, y_train_folds, x_test_folds, y_test_folds


class DrugDataset(torch_geometric.data.Dataset):
    def __init__(self, device, drugs_file, root=None, transform=None, pre_transform=None):
        self.drug_molecules = create_graph_list(drugs_file, device)
        self.transform = transform
        self.pre_transform = pre_transform
        self.__indices__ = None

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.drug_molecules)

    def get(self, idx):
        return self.drug_molecules[idx]

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)

            return data
        else:
            indices = self.index_select(idx)
            chose_data_list = [self.drug_molecules[i] for i in indices]
            batch = Batch.from_data_list(chose_data_list)

            return batch

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(
                    idx.nonzero(as_tuple=False).flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        return indices


class MyDataset(Dataset):
    def __init__(self, device, x_dataset: np.ndarray, y_dataset: np.ndarray,
                 transform=None, target_transform=None):
        # super().__init__(transform, target_transform)

        self.x_dataset = torch.tensor(x_dataset, device=device, dtype=torch.long)
        self.y_dataset = torch.tensor(y_dataset, device=device, dtype=torch.long)

    def __len__(self):
        return self.x_dataset.size()[0]

    def __getitem__(self, idx):
        drug_idx, dis_idx = self.x_dataset[idx]
        label = self.y_dataset[idx]
        sample = {'drug_idx': drug_idx, 'dis_idx': dis_idx, 'label': label}

        return sample
