import torch

from models.CAML import CAML
from utils import divide_into_query_and_support, get_accuracy_from_logits


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, way, shot):
    model.train()
    avg_loss = 0.0
    avg_acc = 0.0
    for i, (X, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        X, labels = X.to(device), labels.to(device)
        suppX, queryX, suppY, queryY = divide_into_query_and_support(X, labels, way, shot)
        queryY = queryY.squeeze(0)

        if isinstance(model, CAML):
            X = torch.cat([suppX, queryX], dim=1).squeeze(0)
            suppY = suppY.squeeze(0)
            logits = model(X, suppY, way=way, shot=shot)
        else:
            logits = model(suppX, suppY, queryX)
            logits = logits.view(queryX.shape[0] * queryX.shape[1], -1)

        loss = criterion(logits, queryY)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_value = loss.item()

        acc = get_accuracy_from_logits(logits, queryY)
        avg_acc += acc
        avg_loss += loss_value
    avg_acc = round(avg_acc / (i + 1), 3)
    avg_loss = round(avg_loss / (i + 1), 3)

    return avg_loss, avg_acc
