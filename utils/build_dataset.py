import pandas as pd
import numpy as np
import torch
import copy
from typing import List
import torch_geometric
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

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


def create_graph_list(filename) -> List[Data]:
    """
    Convert each smiles of drug to graph.
    :return: list of Data object
    """
    data_list = []
    pf = pd.read_csv(filename, sep=',', header=None)
    data = pf.values
    for row in data[1:]:
        smiles = row[3]
        graph = smiles2graph(smiles)
        data_list.append(graph)

    return data_list


def adj_to_edge_list(filename) -> List[List[int]]:
    pf = pd.read_csv(filename, sep=',', header=None)
    adj = pf.values
    n_drugs, n_dis = adj.shape
    edge_list = []

    for drug_id in range(n_drugs):
        for dis_id in range(n_dis):
            weight = adj[drug_id][dis_id]
            if weight != 0:
                edge_list.append([drug_id, dis_id, 1])
            else:
                edge_list.append([drug_id, dis_id, 0])

    return edge_list


class DrugDataset(torch_geometric.data.Dataset):
    def __init__(self, drugs_file, root=None, transform=None, pre_transform=None):
        self.drug_molecules = create_graph_list(drugs_file)
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
            return self.index_select(idx)

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

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        return dataset


class MyDataset(Dataset):
    def __init__(self, drug_sim: str, dis_sim: str, dataset: List[List[int]],
                 transform=None, target_transform=None):
        # super().__init__(transform, target_transform)
        adj_drug = csv_to_ndarray(drug_sim)
        adj_dis = csv_to_ndarray(dis_sim)

        self.graph_drug = adj_to_graph(adj_drug)
        self.graph_dis = adj_to_graph(adj_dis)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        drug_id, dis_id, label = self.dataset[idx]
        sample = [drug_id, dis_id, label]

        return sample


if __name__ == '__main__':
    drug_dataset = DrugDataset('../data/drugs.csv')
