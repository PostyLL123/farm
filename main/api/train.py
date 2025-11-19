import torch
from tqdm import tqdm
from utils.commn import get_local_rank
import numpy as np
device = torch.device(f'cuda:{get_local_rank()}' if torch.cuda.is_available() else 'cpu')


def train_onepart(cfg, model, train_loader, optimizer, scheduler, epoch, time_counter, criterion):
    model.train()
    loss_all = []

    if cfg.model_name == 'TGCN':
        adj = cfg.adj
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)
    for batch_num, (x, y) in enumerate(pbar):
        optimizer.zero_grad() 

        loss = criterion(model(x, adj), y)

        loss.backward()

        optimizer.step()

        loss_all.append(loss.item())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if batch_num % cfg.display_interval == 0:
            cfg.logger.info(
                '[LOCAL_RANK %d Epoch %d Batch %d ] ||| [lr: %.6f] [Loss: %.4f]  ||| LeftTime %s MaxMemory %dMB' %
                (get_local_rank(),
                 epoch,
                 batch_num,
                 optimizer.param_groups[0]['lr'],
                 loss.item(),
                 time_counter.step(epoch + 1, batch_num),
                 torch.cuda.max_memory_allocated(device) / 1024 ** 2))
            
        return  model, optimizer, scheduler, loss_all

def valid(cfg, model, valid_loader, epoch, criterion):
    model.eval()
    loss_all = []
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f'Valid Epoch {epoch}', ncols=100)
        for batch_num, (x, y) in enumerate(pbar):
            outputs = model(x, cfg.adj)
            loss = criterion(outputs, y)
            loss_all.append(loss.item())

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    return loss_all