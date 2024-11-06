#!/usr/bin/env python3
from tqdm import tqdm, trange

from src import const
from src.dataset import balanced_fine_tune_data
# import time
import torch
from torch import nn
import torch.optim as optim
from src.models.arch import get_models


# Balanced train loop
def train_balanced(
        criterion,
        optimizer,
        scheduler,
        balanced_loader,
        model_g,
        model_f_w):
    c = 0
    for epoch in range(const.fine_tune_epoch):
        # start = time.time()
        train_loss = 0.0
        for batch in tqdm(balanced_loader, desc="Training"):
            x, y = batch
            x, y = x.to(const.device), y.to(const.device)
            y_hat_temp, _ = model_g(x, c)
            y_hat_temp_copy = y_hat_temp.detach().clone()
            y_hat_temp_copy = y_hat_temp.to(const.device)
            y_hat, _ = model_f_w(y_hat_temp_copy, c)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach().cpu().item() / len(balanced_loader)

        # end = time.time()
        # epochtime = end / 60 - start / 60
        scheduler.step()

        print(f"Epoch {epoch + 1} / {const.fine_tune_epoch} \
              loss: {train_loss: .2f}")


if __name__ == '__main__':

    balanced_loader = balanced_fine_tune_data()
    criterion = nn.CrossEntropyLoss()

    model_g, model_f_w, model_f_s = get_models()
    optimizer = optim.Adam(
        list(
            model_g.parameters()) +
        list(
            model_f_w.parameters()),
        lr=const.base_lr,
        weight_decay=const.weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)

    model_g.load_state_dict(torch.load(const.MODEL_DIR / 'model_g'))
    model_f_w.load_state_dict(torch.load(const.MODEL_DIR / 'model_f_w'))
    model_f_s.load_state_dict(torch.load(const.MODEL_DIR / 'model_f_s'))
    train_balanced(
        criterion,
        optimizer,
        scheduler,
        balanced_loader,
        model_g,
        model_f_w)
    torch.save(model_g.state_dict(), const.MODEL_DIR / 'model_g')
    torch.save(model_f_w.state_dict(), const.MODEL_DIR / 'model_f_w')
    torch.save(model_f_s.state_dict(), const.MODEL_DIR / 'model_f_s')
