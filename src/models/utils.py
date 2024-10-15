from models import CAML
from models.consts import ModelType


def get_model(type: ModelType, fe_extractor, fe_dim, fe_dtype, train_fe, encoder_size, **kwargs):
    if type == ModelType.CAML:
        model = CAML(fe_extractor, fe_dim, fe_dtype, train_fe, encoder_size, **kwargs)
    elif type == ModelType.PMF:
        raise NotImplementedError
    else:
        raise ValueError(f"Model type {type} not recognized.")
    return model
