import torch
from dotenv import load_dotenv
from pyaml_env import parse_config

import wandb
from models.feature_extractor import get_pretrained_model, get_transform
from models.utils import get_model

load_dotenv()
config = parse_config('environment.yml')

wandb.require("core")

wandb.init(
    project=config["wandb"]["project-name"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_transform, test_transform = get_transform(config["model"]["name"])
    feature_extractor = get_pretrained_model(config["model"]["feature_extractor"])
    model = get_model(config["model"]["name"], feature_extractor, config["model"]["feature_extractor_dim"],
                      config["model"]["fe_dtype"], False, config["model"]["encoder_size"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["lr"])
