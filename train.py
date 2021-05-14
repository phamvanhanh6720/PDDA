import copy

import torch
from tqdm.notebook import tqdm
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score

from model.model import MyModel
from utils.build_dataset import create_dataset, MyDataset, DrugDataset
from utils.build_dataset import adj_to_graph, csv_to_ndarray
from model.loss import WeightedFocalLoss


class Trainer:
    def __init__(self, dataset_file, drug_sim_file, dis_sim_file, drugs_file,lr, n_epoch, dropout, batch_size=128):
        self.lr = lr
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build dataset and dataloader
        dataset = create_dataset(dataset_file)
        self.train_dataset = MyDataset(self.device, dataset['x_train'], dataset['y_train'])
        self.valid_dataset = MyDataset(self.device, dataset['x_valid'], dataset['y_valid'])
        self.test_dataset = MyDataset(self.device, dataset['x_test'], dataset['y_test'])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        drug_adj = csv_to_ndarray(drug_sim_file)
        self.drug_graph: Data = adj_to_graph(drug_adj)
        self.drug_graph = self.drug_graph.to(self.device)

        dis_adj = csv_to_ndarray(dis_sim_file)
        self.dis_graph:Data = adj_to_graph(dis_adj)
        self.dis_graph = self.dis_graph.to(self.device)

        self.drug_molecules = DrugDataset(drugs_file=drugs_file, device=self.device)

        # Build Model, Optimizer, Loss Function
        self.model = MyModel(drug_input_dim=self.drug_graph.num_node_features,
                             dis_input_dim=self.dis_graph.num_node_features, hidden_dim=256,
                             drug_output_dim=128, dis_output_dim=128, num_layers=4,
                             num_layer2=5, dropout=self.dropout)
        self.model = self.model.to(self.device)

        self.loss_fn = WeightedFocalLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train_one_epoch(self):
        self.model.train()
        loss_train = 0

        for step, batch in enumerate(self.train_dataloader):
            drug_idx = batch['drug_idx']
            dis_idx = batch['dis_idx']
            label = batch['label']
            label = label.to(torch.float)

            self.optimizer.zero_grad()
            logit = self.model(self.drug_graph, drug_idx, self.dis_graph, dis_idx, self.drug_molecules[drug_idx])
            logit = torch.squeeze(logit, dim=1)
            loss = self.loss_fn(logit, label)
            loss_train += loss.item()

            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        return loss_train

    def eval_one_epoch(self, dataloader, threshold=0.5):
        self.model.eval()
        total_loss = 0
        acc = 0
        y_label = list()
        y_pred = list()
        for step, batch in enumerate(dataloader):
            drug_idx = batch['drug_idx']
            dis_idx = batch['dis_idx']
            label = batch['label']
            label = label.to(torch.float)
            y_label.append(label.detach().cpu())

            with torch.no_grad():
                logit = self.model(self.drug_graph, drug_idx, self.dis_graph, dis_idx, self.drug_molecules[drug_idx])
                logit = torch.squeeze(logit, dim=1)
                pred = torch.sigmoid(logit)
                y_pred.append(pred.detach().cpu())

            loss = self.loss_fn(logit, label)
            total_loss += loss.item()

        y_pred = torch.cat(y_pred, dim=0)
        y_label = torch.cat(y_label, dim=0)

        fpr, tpr, thresholds = roc_curve(y_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        area_under_curve = auc(fpr, tpr)
        ap = average_precision_score(y_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        return total_loss / self.batch_size, area_under_curve, ap

    def train(self):
        best_model = None
        best_valid_auc = 0
        for epoch in tqdm(range(self.n_epoch)):
            print("============Epoch {}============".format(epoch+1))
            print("Training epoch {}".format(epoch+1))
            loss = self.train_one_epoch()

            print("Evaluating epoch {}".format(epoch+1))
            train_result = self.eval_one_epoch(self.train_dataloader)
            valid_result = self.eval_one_epoch(self.valid_dataloader)

            train_loss, train_auc, train_ap = train_result
            valid_loss, valid_auc, valid_ap = valid_result

            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc

            print('Loss: {:.5f}, Train AUC: {:.4f},  Train AP: {:.4f}, Valid AUC: {:.4f}, Valid AP: {:.4f}'
                  .format(train_loss, train_auc, train_ap, valid_auc, valid_ap))

        print("Training Done")
        print("Evaluating on test dataset")
        test_resul = self.eval_one_epoch(self.test_dataloader)
        _, test_auc, test_ap = test_resul
        print('Test AUC: {:.4f}, Valid AP: {:.4f}'.format(test_auc, test_ap))


if __name__ == '__main__':
    trainer = Trainer('./data/drug_dis.csv', './data/drug_sim.csv', './data/dis_sim.csv', './data/drugs.csv',
                      lr=0.01, n_epoch=60, dropout=0.2, batch_size=128)
    trainer.train()

