from enum import Enum


class ModelType(Enum):
    CAML = 'CAML'
    PMF = 'PMF'
    PMF_FT = 'PMF_Finetune'
    BASELINE_KMEANS = 'BaselineKMeans'
    BASELINE_KNN = 'BaselineKNN'
