#!/bin/sh

module load anaconda/2022.10

module load cuda/11.6

source activate wjw

export PYTHONUNBUFFERED=1

python main_RL.py