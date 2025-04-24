

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
                    suppX, suppY, queryX, way, shot)
                acc = model.calculate_accuracy(query_groups, queryY.cpu())
                loss_value = 0
            else:
                if isinstance(model, CAML):
                    suppX = suppX.squeeze(0)
                    queryX = queryX.squeeze(0)
                    suppY = suppY.squeeze(0)

                    batch_size = queryX.shape[0]

                    unique_classes = suppY.unique()
                    class_indices = {cls.item(): (suppY == cls).nonzero(as_tuple=True)[0] for cls in unique_classes}
                    query_class_indices = {cls.item(): (queryY == cls).nonzero(
                        as_tuple=True)[0] for cls in unique_classes}

                    classes = list(class_indices.keys())
                    assert len(classes) == way, "Support set classes and way mismatch!"

                    group_size = 5
                    num_groups = (way + group_size - 1) // group_size

                    class_logits = torch.full((batch_size, way), -1e9, device=device)

                    for group_idx in range(num_groups):
                        group_classes = classes[group_idx * group_size: (group_idx + 1) * group_size]

                        # Handle small last group
                        padded_classes = group_classes.copy()
                        while len(padded_classes) < group_size:
                            padded_classes.append(padded_classes[-1])

                        # Indices
                        supp_idx = torch.cat([class_indices[c] for c in group_classes], dim=0)
                        query_idx = torch.cat([query_class_indices[c] for c in group_classes], dim=0)

                        group_suppX = suppX[supp_idx]
                        group_suppY = suppY[supp_idx]
                        group_queryX = queryX[query_idx]
                        group_queryY = queryY[query_idx]

                        group_input = torch.cat([group_suppX, group_queryX], dim=0)

                        # Remap labels
                        remap_dict = {orig_class: new_idx for new_idx, orig_class in enumerate(group_classes)}
                        group_suppY_remapped = group_suppY.clone()
                        group_queryY_remapped = group_queryY.clone()

                        for orig, remapped in remap_dict.items():
                            group_suppY_remapped[group_suppY == orig] = remapped
                            group_queryY_remapped[group_queryY == orig] = remapped

                        logits = model(group_input, group_suppY_remapped, way=group_size, shot=shot)

                        # Assign logits back to correct position
                        for j, idx in enumerate(query_idx):
                            for k, orig_class in enumerate(group_classes):
                                class_position = classes.index(orig_class)
                                class_logits[idx, class_position] = logits[j, k]

                    loss = criterion(class_logits, queryY)
                    loss_value = loss.item()
                    acc = get_accuracy_from_logits(class_logits, queryY)
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
