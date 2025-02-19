import torch

from models.baseline import Baseline
from models.CAML import CAML
from models.consts import ModelType
from models.protonet import ProtoNet, ProtoNet_Finetune


def get_model(type: ModelType, fe_extractor, fe_dim=768, fe_dtype=torch.float32,
              train_fe=False, encoder_size="tiny", device="cpu", label_elmes=True, **kwargs):
    if type == ModelType.CAML:
        model = CAML(feature_extractor=fe_extractor, fe_dim=fe_dim, fe_dtype=fe_dtype, train_fe=train_fe,
                     encoder_size=encoder_size, device=device, label_elmes=label_elmes, **kwargs)
    elif type == ModelType.PMF:
        model = ProtoNet(backbone=fe_extractor)
    elif type == ModelType.PMF_FT:
        model = ProtoNet_Finetune(backbone=fe_extractor, **kwargs)
    elif type == ModelType.BASELINE:
        model = Baseline(feature_extractor=fe_extractor, extractor_dim=fe_dim)
    else:
        raise ValueError(f"Model type {type} not recognized.")
    return model
