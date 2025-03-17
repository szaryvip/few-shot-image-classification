from typing import Any

import timm
import torchvision
from transformers import AutoConfig, AutoModel


def get_pretrained_model(model_name: str):
    if "timm" in model_name:
        model = timm.create_model(model_name,
                                  pretrained=True,
                                  num_classes=0).eval()
    elif "MambaVision" in model_name:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda().eval()
    return model


def get_transform_for_timm(model_name: str):
    data_config = timm.data.resolve_model_data_config(model_name)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    test_transform = timm.data.create_transform(**data_config, is_training=False)

    return train_transform, test_transform


def get_transform_for_mamba(model_name: str, model: Any):
    inp_size = (3, 224, 224)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    train_transform = timm.data.transforms_factory.create_transform(
        input_size=inp_size,
        is_training=True,
        mean=config.mean,
        std=config.std,
        crop_mode=config.crop_mode,
        crop_pct=config.crop_pct
    )
    test_transform = timm.data.transforms_factory.create_transform(
        input_size=inp_size,
        is_training=False,
        mean=config.mean,
        std=config.std,
        crop_mode=config.crop_mode,
        crop_pct=config.crop_pct
    )
    return train_transform, test_transform


def get_transform(model_name: str, is_not_pil: bool = False, model: Any = None):
    if "timm" in model_name:
        train_transform, test_transform = get_transform_for_timm(model_name)
    elif "MambaVision" in model_name:
        train_transform, test_transform = get_transform_for_mamba(model_name, model)

    if is_not_pil:
        train_transform = [transform if not isinstance(transform, torchvision.transforms.ToTensor) else torchvision.transforms.Lambda(
            lambda x: x / 255.0) for transform in train_transform.transforms]
        test_transform = [transform if not isinstance(transform, torchvision.transforms.ToTensor) else torchvision.transforms.Lambda(
            lambda x: x / 255.0) for transform in test_transform.transforms]

        train_transform = torchvision.transforms.Compose(train_transform)
        test_transform = torchvision.transforms.Compose(test_transform)

    return train_transform, test_transform
