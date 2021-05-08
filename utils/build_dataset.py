import pandas as pd
import numpy as np
import torch
from typing import List, Union
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


def create_data_list(filename) -> List[Data]:
    """
    Convert each smiles fo drug to graph.
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


class MyDataset(Dataset):
    def __init__(self, drugs_file: str, drug_sim: str, dis_sim: str, dataset: List[List[int]]
                 , transform=None, target_transform=None):
        #super().__init__(transform, target_transform)
        adj_drug = csv_to_ndarray(drug_sim)
        adj_dis = csv_to_ndarray(dis_sim)

        self.graph_drug = adj_to_graph(adj_drug)
        self.graph_dis = adj_to_graph(adj_dis)

        self.drug_molecules = create_data_list(drugs_file)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        drug_id, dis_id, label = self.dataset[idx]
        drug_mol = self.drug_molecules[drug_id]

        sample = [drug_id, dis_id, label]

        return sample


if __name__ == '__main__':
    dataset = adj_to_edge_list('../data/drug_dis.csv')
    drug_file = '../data/drugs.csv'
    drug_sim = '../data/drug_sim.csv'
    dis_sim = '../data/dis_sim.csv'
    mydataset = MyDataset(drug_file, drug_sim, dis_sim, dataset)
    train_loader = DataLoader(mydataset, batch_size=32, shuffle=True)
    for index, data in enumerate(train_loader):
        print(data)
        break

