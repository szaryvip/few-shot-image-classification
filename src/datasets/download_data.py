import shutil

import learn2learn.vision.datasets as l2l
import torch

from datasets.consts import Dataset, DatasetPath, DatasetType


def download_data(dataset_name: Dataset,
                  dataset_type: DatasetType) -> torch.utils.data.Dataset:
    if dataset_name == Dataset.FC100:
        return l2l.FC100(root=DatasetPath.FC100.value,
                         download=True,
                         mode=dataset_type.value)
    elif dataset_name == Dataset.MINI_IMAGENET:
        return l2l.MiniImagenet(DatasetPath.MINI_IMAGENET.value,
                                download=True,
                                mode=dataset_type.value)
    elif dataset_name == Dataset.TIERED_IMAGENET:
        return l2l.TieredImagenet(DatasetPath.TIERED_IMAGENET.value,
                                  download=True,
                                  mode=dataset_type.value)
    elif dataset_name == Dataset.CUB200:
        return l2l.CUBirds200(DatasetPath.CUB200.value,
                              download=True,
                              mode=dataset_type.value)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))


def remove_data(dataset_path: DatasetPath) -> None:
    shutil.rmtree(dataset_path.value)
