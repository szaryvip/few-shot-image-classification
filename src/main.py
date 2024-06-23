from dotenv import load_dotenv
from pyaml_env import parse_config

import wandb

load_dotenv()
config = parse_config('environment.yml')

wandb.require("core")

wandb.init(
    project=config["wandb"]["project-name"]
)
