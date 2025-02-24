#!/bin/sh
# First argument is the model name
# Example: ./experiments/run_laion2b.sh Baseline

fe='timm:vit_huge_patch14_clip_224.laion2b'
fe_dim=1280

. ./experiments/prepare.sh

for dataset in $(echo $DATASETS | sed "s/;/ /g")
do
    echo "Dataset: $dataset"
    python main.py --model=$1 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --use_wandb --epochs=0
done
