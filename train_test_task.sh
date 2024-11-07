#!/bin/bash

# Make appropriate changes to const.py according to the specific experiment 
# and task before running this script 

python -m src.models.train_task
python -m src.models.test
python -m src.models.train_finetune_balanced_dataset
python -m src.models.test