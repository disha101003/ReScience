from src.models.test import test
from src import const
from src.dataset import test_data
import argparse
from pathlib import Path
import torch 
from torch import nn

# For Testing with  pretrained models only
# PRETRAINED_BASE_PATH = BASE_DIR/'models'/'pretrained_models_CiFAR10' for
# CiFAR10
# PRETRAINED_BASE_PATH = BASE_DIR/'models'/'pretrained_models_CiFAR100'for
# CiFAR100
PRETRAINED_BASE_PATH = const.BASE_DIR / 'models' / 'pretrained_models_CiFAR10'

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_fw_path', type=Path, default=PRETRAINED_BASE_PATH / 'cifar10ft4_fw.pth',
                        help='Path to the model file for FW (default: models/pretrained_models_CiFAR10/cifar10ft4_fw.pth)')
    parser.add_argument('--model_g_path', type=Path, default=PRETRAINED_BASE_PATH / 'cifar100ft4_g.pth',
                        help='Path to the model file for G (default: models/pretrained_models_CiFAR10/cifar10ft4_g.pth)')

    # Parse arguments
    args = parser.parse_args()

    testloader = test_data()
    criterion = nn.CrossEntropyLoss()
    model_g = torch.load(args.model_g_path, weights_only=False)
    model_f_w = torch.load(args.model_fw_path, weights_only=False)
    test(criterion, testloader, model_g, model_f_w)

if __name__ == "__main__":
    main()

