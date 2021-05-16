import torch
from tqdm.notebook import tqdm
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score

from model.model import MyModel
from utils.build_dataset import create_dataset, MyDataset, DrugDataset
from utils.build_dataset import adj_to_graph, csv_to_ndarray
from utils.log import write_file
from model.loss import WeightedFocalLoss


class Trainer:

    def __init__(self, dataset_file, drug_sim_file, dis_sim_file, drugs_file, lr, n_epoch, dropout,
                 batch_size=128, k_fold=5):

        self.k_fold = k_fold
        self.lr = lr
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build dataset and dataloader
        self.x_train_folds, self.y_train_folds, self.x_test_folds, self.y_test_folds = create_dataset(dataset_file,
                                                                                                      k_fold=5)

        drug_adj = csv_to_ndarray(drug_sim_file)
        self.drug_graph: Data = adj_to_graph(drug_adj)
        self.drug_graph = self.drug_graph.to(self.device)

        dis_adj = csv_to_ndarray(dis_sim_file)
        self.dis_graph: Data = adj_to_graph(dis_adj)
        self.dis_graph = self.dis_graph.to(self.device)

        self.drug_molecules = DrugDataset(drugs_file=drugs_file, device=self.device)

        # Build Model, Optimizer, Loss Function
        self.loss_fn = WeightedFocalLoss(alpha=0.75)

    def train_one_epoch(self, dataloader, model, optimizer, scheduler):
        model.train()
        loss_train = 0

        for step, batch in enumerate(dataloader):
            drug_idx = batch['drug_idx']
            dis_idx = batch['dis_idx']
            label = batch['label']
            label = label.to(torch.float)

            optimizer.zero_grad()
            logit = model(self.drug_graph, drug_idx, self.dis_graph, dis_idx, self.drug_molecules[drug_idx])
            logit = torch.squeeze(logit, dim=1)
            loss = self.loss_fn(logit, label)
            loss_train += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        return loss_train

    def eval_one_epoch(self, dataloader, model, threshold=0.5):
        model.eval()
        total_loss = 0
        y_label = list()
        y_pred = list()
        for step, batch in enumerate(dataloader):
            drug_idx = batch['drug_idx']
            dis_idx = batch['dis_idx']
            label = batch['label']
            label = label.to(torch.float)
            y_label.append(label.detach().cpu())

            with torch.no_grad():
                logit = model(self.drug_graph, drug_idx, self.dis_graph, dis_idx, self.drug_molecules[drug_idx])
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

        return round(total_loss / self.batch_size, 5), round(area_under_curve, 5), round(ap, 5)

    def train(self, base_root='./weight/'):
        avg_auc = 0
        avg_ap = 0

        for k in range(self.k_fold):
            print("++++++++++++++ This is {}th cross validation ++++++++++++++".format(k + 1))
            torch.cuda.empty_cache()
            train_dataset = MyDataset(self.device, self.x_train_folds[k], self.y_train_folds[k])
            test_dataset = MyDataset(self.device, self.x_test_folds[k], self.y_test_folds[k])

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            model = MyModel(drug_input_dim=self.drug_graph.num_node_features,
                                 dis_input_dim=self.dis_graph.num_node_features, hidden_dim=512,
                                 drug_output_dim=128, dis_output_dim=128, num_layers=4,
                                 num_layer2=5, dropout=self.dropout)
            model = model.to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            file_log = base_root + '{}th.txt'.format(k + 1)

            for epoch in tqdm(range(self.n_epoch)):
                print("============Epoch {}============".format(epoch + 1))
                print("Training epoch {}".format(epoch + 1))
                loss = self.train_one_epoch(train_dataloader, model, optimizer, scheduler)

                print("Evaluating epoch {}".format(epoch + 1))
                train_result = self.eval_one_epoch(train_dataloader, model)
                test_result = self.eval_one_epoch(test_dataloader, model)

                train_loss, train_auc, train_ap = train_result
                test_loss, test_auc, test_ap = test_result

                write_file(file_log, epoch + 1, train_result, test_result)
                print('Loss: {:.5f}, Train AUC: {:.4f},  Train AP: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'
                      .format(train_loss, train_auc, train_ap, test_auc, test_ap))

            print("Evaluating on test dataset")
            test_result = self.eval_one_epoch(test_dataloader, model)
            _, test_auc, test_ap = test_result
            avg_ap += test_ap
            avg_auc += test_auc
            print('Test AUC: {:.4f}, Test AP: {:.4f} on {}th cross validation'.format(test_auc, test_ap, k + 1))

        print('Avg AUC: {:.4f}, Avg AP: {:.4f}'.format(avg_auc / 5, avg_ap / 5))


if __name__ == '__main__':
    trainer = Trainer('./data/drug_dis.csv', './data/drug_sim.csv', './data/dis_sim.csv', './data/drugs.csv',
                      lr=0.01, n_epoch=60, dropout=0.3, batch_size=128)
    trainer.train(base_root='./weight/5_fold_v2/')
