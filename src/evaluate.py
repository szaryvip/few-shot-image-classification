

import torch

from models.baseline import Baseline
from models.CAML import CAML
from utils import divide_into_query_and_support, get_accuracy_from_logits


def eval_func(model, dataloader, criterion, device, way, shot):
    model.eval()
    avg_acc = 0.0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (X, labels) in enumerate(dataloader):
            X, labels = X.to(device), labels.to(device)
            suppX, queryX, suppY, queryY = divide_into_query_and_support(X, labels, way, shot)
            queryY = queryY.squeeze(0)

            if isinstance(model, Baseline):
                supp_groups, query_groups, supp_features, query_features = model.get_groups_and_features(
                    suppX, queryX, way)
                acc = model.calculate_accuracy(query_groups, queryY.cpu())
                loss_value = 0
            else:
                if isinstance(model, CAML):
                    X = torch.cat([suppX, queryX], dim=1).squeeze(0)
                    suppY = suppY.squeeze(0)
                    logits = model(X, suppY, way=way, shot=shot)
                else:
                    logits = model(suppX, suppY, queryX)
                    logits = logits.view(queryX.shape[0] * queryX.shape[1], -1)

                loss = criterion(logits, queryY)
                loss_value = loss.item()
                acc = get_accuracy_from_logits(logits, queryY)
            avg_acc += acc
            avg_loss += loss_value
    avg_acc = round(avg_acc / (i + 1), 3)
    avg_loss = round(avg_loss / (i + 1), 3)
    return avg_loss, avg_acc
