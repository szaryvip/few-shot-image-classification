import timm
import torchvision


def get_pretrained_model(model_name: str):
    model = timm.create_model(model_name,
                              pretrained=True,
                              num_classes=0).eval()
    return model


def get_transform(model_name: str):
    data_config = timm.data.resolve_model_data_config(model_name)
    train_transform = timm.data.create_transform(**data_config, is_training=False)
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    train_transform = [transform if not isinstance(transform, torchvision.transforms.ToTensor) else torchvision.transforms.Lambda(
        lambda x: x / 255.0) for transform in train_transform.transforms]
    test_transform = [transform if not isinstance(transform, torchvision.transforms.ToTensor) else torchvision.transforms.Lambda(
        lambda x: x / 255.0) for transform in test_transform.transforms]

    train_transform = torchvision.transforms.Compose(train_transform)
    test_transform = torchvision.transforms.Compose(test_transform)
    return train_transform, test_transform
