from argparse import ArgumentParser

import torch

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["CAML", "PMF", "PMF_Finetune"],
        help="Specify the model type for experiment. Options: ['CAML', 'PMF', 'PMF_Finetune']"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["miniImageNet", "tieredImageNet"],
        help="Specify the dataset for experiment. Options: ['miniImageNet', 'tieredImageNet']"
    )

    parser.add_argument(
        "--way",
        type=int,
        default=5,
        help="Specify the number of classes in a task. Default: 5"
    )

    parser.add_argument(
        "--shot",
        type=int,
        default=5,
        help="Specify the number of support examples per class. Default: 5"
    )

    parser.add_argument(
        "--number_of_tasks",
        type=int,
        default=20,
        help="Specify the number of tasks for training and evaluation. Default: 20"
    )

    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="timm/vit_small_patch16_224.dino",
        help="Specify the feature extractor for the model. Default: 'timm/vit_small_patch16_224.dino'"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Specify the number of epochs for training. If 0, only evaluation is performed. Default: 10"
    )

    parser.add_argument(
        "--encoder_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large", "convnext", "laion", "resnet34", "huge"],
        help="Specify the size of the encoder in CAML model. Options: ['tiny', 'small', 'base', 'large', 'convnext', 'laion', 'resnet34', 'huge']. Default: 'tiny'"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable logging to Weights & Biases."
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="few-shot-learning",
        help="Specify the Weights & Biases project name. Default: 'few-shot-learning'"
    )

    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Disable CUDA."
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
