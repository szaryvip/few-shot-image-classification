import torch


def divide_into_query_and_support(X, labels, way, shot):
    """Input shape:
    X: [B, way*(shot+1), C, H, W]
    labels: [B, way*(shot+1)]
    Output: suppX, queryX, suppY, queryY
    with the same shapes as the input."""
    B, _, C, H, W = X.shape
    groupedX = X.view(B, way, shot+1, C, H, W)
    groupedLabels = labels.view(B, way, shot+1)
    suppX = groupedX[:, :, :shot, :, :, :].reshape(B, way*shot, C, H, W)
    queryX = groupedX[:, :, shot, :, :, :].reshape(B, way, C, H, W)
    suppY = groupedLabels[:, :, :shot].reshape(B, way*shot)
    queryY = groupedLabels[:, :, shot].reshape(B, way)
    return suppX, queryX, suppY, queryY


def get_accuracy_from_logits(logits, labels):
    """Compute the accuracy given the logits and the labels."""
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()*100
    return round(acc, 3)


def count_learnable_params(model):
    """Count the number of learnable parameters in a pytorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_non_learnable_params(model):
    """Count the number of non-learnable parameters in a pytorch model."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)
