import timm


def get_pretrained_model(model_name: str):
    model = timm.create_model(model_name,
                              pretrained=True,
                              num_classes=0).eval()
    return model


def get_transform(model_name: str):
    data_config = timm.data.resolve_model_data_config(model_name)
    train_transform = timm.data.create_transform(**data_config, is_training=False)
    test_transform = timm.data.create_transform(**data_config, is_training=False)
    return train_transform, test_transform
