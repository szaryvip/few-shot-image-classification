#!/bin/sh
# First argument is the model name
# Example: ./experiments/run_mamba.sh BaselineKMeans

fe='nvidia/MambaVision-B-1K'
fe_dim=640

. ./experiments/prepare.sh

for dataset in $(echo $DATASETS | sed "s/;/ /g")
do
    echo "Dataset: $dataset"
    if [ $1 = "CAML" ]; then
        echo "CAML not supported for MambaVision"
    else
        python main.py --model=$1 --dataset=$dataset --feature_extractor=$fe --fe_dim=$fe_dim --use_wandb --epochs=0
    fi
done
