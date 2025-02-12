from enum import Enum


class DatasetType(Enum):
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'


class Dataset(Enum):
    FC100 = 'fc100'
    MINI_IMAGENET = 'mini-imagenet'
    TIERED_IMAGENET = 'tiered-imagenet'  # 12GB! LARGE AND LIKE MINI-IMAGENET
    CUB200 = 'cub200'  # 1.1GB
    VGGFLOWER102 = 'vggflower102'  # 330MB
    FGVC_AIRCRAFT = 'fgvc-aircraft'  # 2.75GB
    FGVC_FUNGI = 'fgvc-fungi'  # 12.9GB! READ ERROR
    DESC_TEXTURES = 'describable-textures'  # 600MB


class DatasetPath(Enum):
    FC100 = 'datasets/fc100'
    MINI_IMAGENET = 'datasets/mini-imagenet'
    TIERED_IMAGENET = 'datasets/tiered-imagenet'
    CUB200 = 'datasets/cub200'
    VGGFLOWER102 = 'datasets/vggflower102'
    FGVC_AIRCRAFT = 'datasets/fgvc-aircraft'
    FGVC_FUNGI = 'datasets/fgvc-fungi'
    DESC_TEXTURES = 'datasets/describable-textures'
