from enum import Enum


class DatasetType(Enum):
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'


class Dataset(Enum):
    FC100 = 'fc100'
    MINI_IMAGENET = 'mini-imagenet'
    TIERED_IMAGENET = 'tiered-imagenet'
    CUB200 = 'cub200'


class DatasetPath(Enum):
    FC100 = 'datasets/fc100'
    MINI_IMAGENET = 'datasets/mini-imagenet'
    TIERED_IMAGENET = 'datasets/tiered-imagenet'
    CUB200 = 'datasets/cub200'
