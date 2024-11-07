#!/usr/bin/env python3
from tqdm import tqdm

from src import const
from src.dataset import test_data
import torch
from torch import nn

from src.models.arch import get_models

"""
Evaluates the model on a test dataset and computes the loss and accuracy.

Args:
    criterion: Cross-entropy loss function
    testloader: DataLoader for the test dataset.
    model_g: The first part of the ViT model, which includes
        the embedding layers and the first two encoder blocks.
    model_f_w: The second part of the ViT model, which includes
        the remaining encoder blocks and the MLP head.

"""


def test(criterion, testloader, model_g, model_f_w):
    c = 0
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(testloader, desc="Testing"):
            x, y = batch
            x, y = x.to(const.device), y.to(const.device)
            y_hat_temp, _ = model_g(x, c)
            y_hat_temp_copy = y_hat_temp.detach().clone()
            y_hat_temp_copy = y_hat_temp.to(const.device)
            y_hat, _ = model_f_w(y_hat_temp_copy, c)

            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(testloader)

            correct += torch.sum(torch.argmax(y_hat, dim=1)
                                 == y).detach().cpu().item()
            total += len(x)

        accuracy = correct / total * 100
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {accuracy}%")


if __name__ == '__main__':

    testloader = test_data()
    criterion = nn.CrossEntropyLoss()
 
    model_g, model_f_w, model_f_s = get_models()
    model_g.load_state_dict(torch.load(const.MODEL_DIR / 'model_g'))
    model_f_w.load_state_dict(torch.load(const.MODEL_DIR / 'model_f_w'))
    test(criterion, testloader, model_g, model_f_w)
