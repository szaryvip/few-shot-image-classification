#!/bin/sh
# First argument is the model name
# Example: ./experiments/run_laion2b.sh BaselineKMeans

fe='timm:vit_huge_patch14_clip_224.laion2b'
fe_dim=1280

. ./experiments/prepare.sh

for dataset in $(echo $DATASETS | sed "s/;/ /g")
do
    echo "Dataset: $dataset"
    for way in $(echo $WAYS | sed "s/;/ /g")
    do
        echo "Way: $way"
        if [ $1 = "CAML" ]; then
            python main.py --model=$1 --pretrained_path=$2 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --encoder_size=laion --use_wandb --epochs=0 --way=$way
        else
            python main.py --model=$1 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --use_wandb --epochs=0 --way=$way
        fi
    done
done
