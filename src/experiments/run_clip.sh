#!/bin/sh
# First argument is the model name
# Example: ./experiments/run_clip.sh Baseline

fe='timm:vit_base_patch16_clip_224.openai'
fe_dim=768

. ./experiments/prepare.sh

for dataset in $(echo $DATASETS | sed "s/;/ /g")
do
    echo "Dataset: $dataset"
    if [ $1 = "CAML" ]; then
        python main.py --model=$1 --pretrained_path=$2 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --encoder_size=large --use_wandb --epochs=0 
    else
        python main.py --model=$1 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --use_wandb --epochs=0
    fi
done
