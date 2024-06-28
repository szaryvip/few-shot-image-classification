import timm


def get_pretrained_model(model_name: str):
    model = timm.create_model(model_name,
                              pretrained=True,
                              num_classes=0).eval()
    return model
