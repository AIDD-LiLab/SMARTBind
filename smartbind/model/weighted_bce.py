from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight):
        super(WeightedBCELoss, self).__init__()
        self.positive_weight = positive_weight

    def forward(self, input, target):
        loss = nn.BCELoss(reduction='none')(input, target)
        weighted_loss = loss * (target * self.positive_weight + (1 - target))
        return weighted_loss.mean()
