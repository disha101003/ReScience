#!/usr/bin/env python3
from tqdm import tqdm, trange

from src import const
from src.dataset import train_data
import torch.nn.functional as F
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import os
import pickle

from src.models.arch import get_models, D_buffer


def train_task(task_index, trainloader, sem_mem, model_g, model_f_w, model_f_s, criterion, optimizer):
    c = 0
    # training loop of BiRT
    start = int(time.time()/60) # task training start time
    for epoch in range(const.epochs):
        train_loss = 0.0
        task_sem_mem_list = []

        for batch_idx, batch in enumerate(tqdm((trainloader), desc=f"Epoch {epoch + 1} / {const.epochs} in training", leave=False)):
            x, y = batch
            x, y = x.to(const.device), y.to(const.device)
            y_hat_g,_ = model_g(x,c)                      # get output from g()
            y_hat_temp_copy = y_hat_g.detach().clone()  # create a detached copy to be processed further

            # store output from g() along with labels in episodic memory
            y_hat_temp_mem = y_hat_temp_copy.clone()
            y_mem = y
            task_sem_mem_list.append((y_hat_temp_mem, y_mem))

            # controls sampled from a normal distribution to control different noises introduced in BiRT
            alpha_t_comp, alpha_a_comp, alpha_s_comp, alpha_e_comp = np.random.uniform(0,1,4)

            # sampling mini batch from episodic memory
            if (sem_mem.is_empty() == False):
                r, r_y = sem_mem.get_batch()
                r = r.to((const.device))
                r_y = r_y.to(const.device)

                # implementing label noise
                if(alpha_t_comp < const.alpha_t):
                    num_change = int(const.percentage_change / 100 * const.batch_size)
                    indices_change_r_y = torch.randperm(len(y))[:num_change].to(const.device)
                    r_y_changed = torch.randint(0,const.num_classes,(num_change,)).to(const.device)
                    r_y[indices_change_r_y] = r_y_changed

                # implementing attention noise
                if(alpha_a_comp < const.alpha_a):
                    c = 1

                r_y_working,_ = model_f_w(r,c)
                r_y_semantic,_ = model_f_s(r,c)

                # adding noise to logits of semantic/episodic memory
                if(alpha_s_comp < const.alpha_s):
                    r_y_semantic =  r_y_semantic + torch.rand(r_y_semantic.size()).to(const.device)*const.std + const.mean

            y_hat_temp = y_hat_temp_copy.to(const.device)
            y_working,_ = model_f_w(y_hat_temp,c)
            y_semantic,_ = model_f_s(y_hat_temp, c)


            # computing loss
            if(task_index == 0):
                loss_representation = criterion(y_working,y)   # representation loss
            else:
                loss_representation = criterion(y_working,y) + const.alpha_loss_rep*criterion(model_f_w(r, c)[0], r_y)  # loss

            if(task_index == 0):
                loss_consistency_reg = const.beta_1_loss * torch.norm(y_working - y_semantic, p=2)**2
            else:
                loss_consistency_reg = const.beta_1_loss * torch.norm(y_working - y_semantic, p=2)**2  + const.beta_2_loss*torch.norm( r_y_working-r_y_semantic, p = 2)**2  # consistency regulation noise

            loss = loss_representation + const.rho_loss_cr * loss_consistency_reg   # total loss
            loss = loss/const.accum_iter
            loss.backward(retain_graph = True)

            # print(loss)
            if ((batch_idx + 1) % const.accum_iter == 0) or (batch_idx + 1 == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()

            # interpolating parameters of epiodic memory at intervals
            if(alpha_e_comp < const.alpha_e and task_index > 0):
                for params1, params2 in zip(model_f_s.parameters(), model_f_w.parameters()):
                    interpolated_params = const._gamma * params1.data + (1 - const._gamma) * params2.data
                    params1.data.copy_(interpolated_params)

            train_loss += loss.detach().cpu().item() /len(trainloader)

        # Printing average loss per epoch
        print(f"Epoch {epoch + 1}/{const.epochs} loss: {train_loss:.2f}")

    # copying f_w() paramerters to f_s() for first task
    if task_index == 0:
        for params1, params2 in zip(model_f_s.parameters(), model_f_w.parameters()):
            interpolated_params = params2.data
            params1.data.copy_(interpolated_params)

    end = int(time.time()/60) # task training end time
    task_train_time = end - start
    print(f"Task {task_index} done in {task_train_time} mins")
    start = int(time.time()/60)

    sem_mem.update(task_sem_mem_list, task_index)
    
    end = int(time.time()/60) # task training end time
    mem_update_time = end - start
    print(f"Memory {task_index} updated in {mem_update_time} mins")
    torch.save(model_g.state_dict(),const.MODEL_DIR/'model_g' )
    torch.save(model_f_w.state_dict(),const.MODEL_DIR/'model_f_w' )
    torch.save(model_f_s.state_dict(),const.MODEL_DIR/'model_f_s' )
    return sem_mem
        

if __name__ == '__main__':
  
    model_g, model_f_w, model_f_s = get_models()

    model_g_path = os.path.join(const.MODEL_DIR/'model_g')
    model_f_w_path = os.path.join(const.MODEL_DIR/'model_f_w')
    model_f_s_path = os.path.join(const.MODEL_DIR/'model_f_s')
    if (os.path.exists(model_g_path)):
        model_g.load_state_dict(torch.load(const.MODEL_DIR/'model_g'))
    if (os.path.exists(model_f_w_path)):
        model_f_w.load_state_dict(torch.load(const.MODEL_DIR/'model_f_w'))
    if (os.path.exists(model_f_s_path)):
        model_f_s.load_state_dict(torch.load(const.MODEL_DIR/'model_f_s'))

    memory_file_path = os.path.join(const.MEMORY_DIR, 'memory.pt')
    if(os.path.exists(memory_file_path)):
       sem_mem = torch.load(memory_file_path)
    else:
        sem_mem = D_buffer(const.sem_mem_length,const.batch_size,const.num_classes, const.tasks )   
    task_list = train_data()
    optimizer = optim.Adam(list(model_g.parameters()) + list(model_f_w.parameters()), lr=const.base_lr, weight_decay=const.weight_decay)
    criterion = nn.CrossEntropyLoss()

  
    sem_mem = train_task(const.task_num, task_list[const.task_num], sem_mem, model_g, model_f_w, model_f_s,criterion, optimizer )

    torch.save(sem_mem, memory_file_path)

    torch.save(model_g.state_dict(),const.MODEL_DIR/'model_g' )
    torch.save(model_f_w.state_dict(),const.MODEL_DIR/'model_f_w' )
    torch.save(model_f_s.state_dict(),const.MODEL_DIR/'model_f_s' )
 