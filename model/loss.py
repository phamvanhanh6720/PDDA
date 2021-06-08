import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_weights_inverse_num_of_samples(samples_per_cls, device, power=1):
    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * 2

    weights_for_samples = torch.tensor(weights_for_samples, device=device)

    return weights_for_samples


class WeightedBCELoss(nn.Module):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.samples_per_cls = [18416, 142446]
        weight_class = float(self.samples_per_cls[1] / self.samples_per_cls[0])
        self.weight_class = torch.tensor([weight_class], dtype=torch.float, device=self.device)

        self.weight_samples = get_weights_inverse_num_of_samples(samples_per_cls=self.samples_per_cls,
                                                                 device=self.device)

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.weight_class, reduction='none')
        targets = targets.to(torch.long)
        weight_samples = self.weight_samples.gather(0, targets.data.view(-1))
        weight_samples = weight_samples.to(self.device)

        loss_ce = weight_samples * bce_loss

        return loss_ce.mean()


class WeightedFocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.alpha = torch.tensor([alpha, 1-alpha], device=self.device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.to(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = at.to(self.device)
        pt = torch.exp(-bce_loss)
        focal_loss = at*(1-pt)**self.gamma * bce_loss

        return focal_loss.mean()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    input = torch.randn(5, requires_grad=True, device=device)
    target = torch.empty(5, device=device).random_(2)
    print(input.device)
    criterion = WeightedBCELoss()
    loss = criterion(input, target)
    print(loss)