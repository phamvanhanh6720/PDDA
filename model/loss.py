import torch.nn as nn
import torch
import torch.nn.functional as F


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
    loss_fn = WeightedFocalLoss()
    loss = loss_fn(input, target)

    print(loss)