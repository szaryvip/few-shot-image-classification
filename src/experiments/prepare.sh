#!/bin/sh

pip install -r ../requirements.txt

export DATASETS="fc100;mini-imagenet;cub200;vggflower102;describable-textures"
export WAYS="5;10;20;40;70;100"
