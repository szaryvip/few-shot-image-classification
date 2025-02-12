import shutil
from typing import Any

import learn2learn.vision.datasets as l2l
import torch

from datasets.consts import Dataset, DatasetPath, DatasetType


def download_data(dataset_name: Dataset,
                  dataset_type: DatasetType, transform: Any | None = None) -> torch.utils.data.Dataset:
    if dataset_name == Dataset.FC100:
        return l2l.FC100(root=DatasetPath.FC100.value,
                         download=True,
                         mode=dataset_type.value,
                         transform=transform)
    elif dataset_name == Dataset.MINI_IMAGENET:
        return l2l.MiniImagenet(DatasetPath.MINI_IMAGENET.value,
                                download=True,
                                mode=dataset_type.value,
                                transform=transform)
    elif dataset_name == Dataset.TIERED_IMAGENET:
        return l2l.TieredImagenet(DatasetPath.TIERED_IMAGENET.value,
                                  download=True,
                                  mode=dataset_type.value,
                                  transform=transform)
    elif dataset_name == Dataset.CUB200:
        return l2l.CUBirds200(DatasetPath.CUB200.value,
                              download=True,
                              mode=dataset_type.value,
                              transform=transform)
    elif dataset_name == Dataset.VGGFLOWER102:
        return l2l.VGGFlower102(DatasetPath.VGGFLOWER102.value,
                                download=True,
                                mode=dataset_type.value,
                                transform=transform)
    elif dataset_name == Dataset.FGVC_AIRCRAFT:
        return l2l.FGVCAircraft(DatasetPath.FGVC_AIRCRAFT.value,
                                download=True,
                                mode=dataset_type.value,
                                transform=transform)
    elif dataset_name == Dataset.FGVC_FUNGI:
        return l2l.FGVCFungi(DatasetPath.FGVC_FUNGI.value,
                             download=True,
                             mode=dataset_type.value,
                             transform=transform)
    elif dataset_name == Dataset.DESC_TEXTURES:
        return l2l.DescribableTextures(DatasetPath.DESC_TEXTURES.value,
                                       download=True,
                                       mode=dataset_type.value,
                                       transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))


def remove_data(dataset_path: DatasetPath) -> None:
    shutil.rmtree(dataset_path.value)
