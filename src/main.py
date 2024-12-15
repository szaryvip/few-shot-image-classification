import time
from argparse import ArgumentParser

import torch
from tqdm import tqdm

import wandb
from datasets.consts import Dataset, DatasetType
from datasets.download_data import download_data
from datasets.get_data_loader import get_data_loader
from evaluate import eval_func
from models.feature_extractor import get_pretrained_model, get_transform
from scheduler import WarmupCosineDecayScheduler
from train import train_epoch
from utils import count_learnable_params, count_non_learnable_params


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
        choices=["mini-imagenet", "tiered-imagenet"],
        help="Specify the dataset for experiment. Options: ['mini-imagenet', 'tiered-imagenet']"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_cuda:
        device = torch.device("cpu")

    print("Downloading and loading the feature extractor...")
    feature_extractor = get_pretrained_model(args.feature_extractor)
    train_transform, test_transform = get_transform(args.feature_extractor)

    print("Downloading and loading the dataset...")
    if args.dataset not in Dataset.__members__:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    train = download_data(args.dataset, DatasetType.TRAIN, transform=train_transform)
    valid = download_data(args.dataset, DatasetType.VAL, transform=test_transform)
    test = download_data(args.dataset, DatasetType.TEST, transform=test_transform)

    print("Preparing DataLoader...")
    train_loader = get_data_loader(train, args.way, args.shot, args.number_of_tasks, True)
    valid_loader = get_data_loader(valid, args.way, args.shot, args.number_of_tasks, True)
    test_loader = get_data_loader(test, args.way, args.shot, args.number_of_tasks, True)

    print("Preparing the model...")
    criterion = torch.nn.CrossEntropyLoss()

    # TODO prepare model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    learnable_params = count_learnable_params(model)
    non_learnable_params = count_non_learnable_params(model)
    print(f"Learnable parameters: {learnable_params}")
    print(f"Non-learnable parameters: {non_learnable_params}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config={})  # TODO config

    print("Training the model...")
    if args.epochs > 0:
        lr_min = 1e-6
        scheduler = WarmupCosineDecayScheduler(optimizer, args.epochs//10, args.epochs, lr_min)
        best_val_acc = 0
        for epoch in tqdm.tqdm(range(args.epochs)):
            epoch_start = time.time()
            avg_loss, avg_acc = train_epoch(model, train_loader, optimizer, scheduler,
                                            criterion, device, args.way, args.shot)
            train_epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} - Loss: {avg_loss}, Acc: {avg_acc}, Time: {train_epoch_time}")
            if args.use_wandb:
                wandb.log({"train_acc": avg_acc, "train_loss": avg_loss, "train_epoch_time": train_epoch_time})

            torch.save(model.state_dict(), "latest_model.pth")

            avg_loss, avg_acc = eval_func(model, valid_loader, criterion, device, args.way, args.shot)
            full_epoch_time = time.time() - epoch_start
            print(f"Validation - Loss: {avg_loss}, Acc: {avg_acc}, Time: {full_epoch_time}")
            if args.use_wandb:
                wandb.log({"valid_acc": avg_acc, "valid_loss": avg_loss, "full_epoch_time": full_epoch_time})

            best_val_acc = max(best_val_acc, avg_acc)

            if avg_acc >= best_val_acc:
                torch.save(model.state_dict(), "best_model.pth")

        model.load_state_dict(torch.load("best_model.pth"), weights_only=True)

    print("Evaluating the model...")
    eval_start = time.time()
    avg_loss, avg_acc = eval_func(model, test_loader, criterion, device, args.way, args.shot)
    eval_time = time.time() - eval_start
    print(f"Validation - Loss: {avg_loss}, Acc: {avg_acc}, Time: {eval_time}")

    if args.use_wandb:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()
