import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()

        gcn_conv_first = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        gcn_conv_last = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        gcn_conv_mid = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.convs = torch.nn.ModuleList([gcn_conv_first]
                                         + [gcn_conv_mid for _ in range(num_layers-2)] + [gcn_conv_last])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=hidden_dim) for _ in range(num_layers - 1)])
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = self.convs[0](x, edge_index)
        out = self.bns[0](out)
        out = F.relu(out, inplace=False)
        out = F.dropout(out, self.dropout)
        for layer in range(1, len(self.convs) - 1, 1):
            out = self.convs[layer](out, edge_index)
            out = self.bns[layer](out)
            out = F.relu(out, inplace=False)
            out = F.dropout(out, self.dropout)

        out = self.convs[-1](out, edge_index)
        if self.return_embeds:
            return out
        else:
            out = self.softmax(out)
            return out


class GCNGraph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCNGraph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        self.gnn_node = GCN(hidden_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=True)

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):
        # returns a batched output tensor for each graph.
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = self.gnn_node(embed, edge_index)
        # out = global_mean_pool(out, batch)
        out = global_add_pool(out, batch)

        return out


class Conv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_features):
        super(Conv1DBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        out = self.conv1d(x)
        out = self.pool(out)
        out = self.bn(out)
        out = F.relu(out, inplace=False)

        return out


class MyModel(torch.nn.Module):
    def __init__(self, drug_input_dim, dis_input_dim, hidden_dim, drug_output_dim,dis_output_dim,
                 num_layers, num_layer2, dropout):
        super(MyModel, self).__init__()

        self.drug_model = GCN(input_dim=drug_input_dim, hidden_dim=hidden_dim, output_dim=drug_output_dim,
                              num_layers=num_layers, dropout=dropout, return_embeds=True)
        self.dis_model = GCN(input_dim=dis_input_dim, hidden_dim=hidden_dim, output_dim=dis_output_dim,
                             num_layers=num_layers, dropout=dropout, return_embeds=True)

        self.drug_molecular = GCNGraph(hidden_dim=hidden_dim, output_dim=drug_output_dim, num_layers=num_layer2,
                                       dropout=dropout)

        self.conv_block_1 = Conv1DBlock(in_channels=3, out_channels=6,
                                        kernel_size=3, stride=1, padding=1, num_features=6)

        self.conv_block_2 = Conv1DBlock(in_channels=6, out_channels=6,
                                        kernel_size=3, stride=1, padding=1, num_features=6)

        self.linear_1 = nn.Linear(in_features=192, out_features=128)
        self.linear_2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, drug_data: Data, drug_idx, dis_data: Data, dis_idx, batched_mol: Batch):
        drug_feature_1 = self.drug_model(drug_data.x, drug_data.edge_index)
        dis_feature = self.dis_model(dis_data.x, dis_data.edge_index)

        drug_feature_1 = drug_feature_1[drug_idx]
        dis_feature = dis_feature[dis_idx]

        drug_feature_2 = self.drug_molecular(batched_mol)

        fea = torch.cat((drug_feature_1, drug_feature_2, dis_feature), dim=1)
        fea = torch.reshape(fea, (fea.size()[0], 3, -1))

        # N * 3 * 128
        fea = self.conv_block_1(fea)
        # N * 6 * 64
        fea = self.conv_block_2(fea)
        # N * 6 * 32
        fea = fea.view(fea.size()[0], -1)
        # N * 192
        fea = self.linear_1(fea)
        # N * 128
        fea = F.relu(fea, inplace=False)
        fea = self.linear_2(fea)

        return fea
