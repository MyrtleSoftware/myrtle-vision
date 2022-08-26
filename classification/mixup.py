import numpy as np
import torch

def mixup_data(X, y, alpha=1):
    """Implement mixup.

    Returns mixed inputs, pairs of targets, and lambda,
    which represents fraction of mixup.

    https://arxiv.org/abs/1710.09412
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = X.size()[0]
    index = torch.randperm(batch_size).cuda() if torch.cuda.is_available() else torch.randperm(batch_size)

    mixed_X = lam*X + (1-lam)*X[index, :]
    y_1, y_2 = y, y[index]

    return mixed_X, y_1, y_2, lam

def mixup_criterion(criterion, pred, y_1, y_2, lam):
    return lam*criterion(pred, y_1) + (1-lam)*criterion(pred, y_2)