import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import importlib
import logging
import sys
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
print(f"Using GPU: {use_gpu}")

project_rot = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_rot not in sys.path:
    sys.path.append(project_rot)

from utils.commn import save_config, __mkdir__, distributed, get_local_rank, TimeCounter
from utils.normalizer import DataNormalizer, create_normalizer, save_normalizer, load_normalizer
from utils.dataset import WindFarmDataset
from nowcasting.main.model import build_optimizer, build_model
from nowcasting.utils.loss_func import custom_loss
from nowcasting.utils.checkpoint import save_checkpoint, load_checkpoint
from main.api.train import train_onepart, valid

def __init__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='TGCN')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--project', type=str, default='farm_predict', help='the farm turbines windspeed and direciton predict')

    args = parser.parse_args()

    config_file = 'main.config.' + args.config
    try:
        xconfig = importlib.import_module(config_file)
    except ImportError:
        print(f'makesure {config_file} exsited')
        sys.exit(1)

    config_save_path = os.path.join(xconfig.workdir, args.project, args.config)
    save_config(config_file, config_save_path)

    xconfig.logger.info(f"[WORK DIR]: {xconfig.workdir}")
    xconfig.logger.info(f'[Configs]: {config_file}')
    
    torch.manual_seed(xconfig.rand_seed)
    np.random.seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)

    if use_gpu:
        torch.cuda.manual_seed_all(xconfig.rand_seed)

    return xconfig

def draw_loss(loss_all,  epoch, cfg, name, val_loss_epochs=None):
    plt.figure()
    plt.plot(range(len(loss_all)), loss_all, label='train')
    if val_loss_epochs is not None:
        plt.plot(range(len(val_loss_epochs)), val_loss_epochs, label='valid')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.title(f'Training Loss')
    s_name = os.path.join(cfg.workdir, cfg.project, cfg.model_name, 'logs', 'imgs', f'loss_img_{name}.png')
    __mkdir__(s_name)
    plt.savefig(s_name)
    plt.close()
    cfg.logger.info(f'[Save Loss imgs Successed Epoch {epoch} ] ||| {s_name}]')


def load_normalize(args):
    logger = args.logger
    if os.path.exists(args.target_normalizer_path) and os.path.exists(args.feature_normalizer_path):
        logger.info('Loading existing normalizers...')
        target_normalizer = load_normalizer(args.target_normalizer_path)
        feature_normalizer = load_normalizer(args.feature_normalizer_path)
        logger.info(f'Loaded target normalizer from {args.target_normalizer_path} and feature normalizer from {args.feature_normalizer_path}')

    else:
        dataloader = np.load(args.data_path, allow_pickle=True)
        data = dataloader['data']
        feature_order = dataloader['feature_order']

        total_time_steps, num_nodes, num_features = data.shape
        logger.info(f'data loaded, {total_time_steps} steps, {num_nodes} turbines, {num_features} features')

        train_split_idx = int(total_time_steps*args.train_split_ratio)
        valid_split_idx = int(total_time_steps*(args.train_split_ratio + args.valid_split_ratio))

        train_data = data[:train_split_idx]

        logger.info(f'Train data: {train_data.shape}')

        logger.info(f'get target: {args.target_feature}')

        target_idx = np.where(feature_order == args.target_feature)[0][0]

        #NOTE whether to use all features or exclude the target feature as input
        feature_indices = [i for i, col in enumerate(feature_order) if col != args.target_feature]
        feature_col = [col for col in feature_order if col != args.target_feature]

        #load or generte norlaizers
        logger.info(f"normalize data using {args.normalizer_method} method")
        logger.info('Creating and fitting normalizers...')
        try:
            target_normalizer, feature_normalizer = DataNormalizer(method=args.normalizer_method), DataNormalizer(method=args.normalizer_method)
            T, N, F = train_data.shape

            target_data_to_fit = train_data[..., target_idx].reshape(T*N, 1)
            target_normalizer.fit(target_data_to_fit)
            logger.info('Fitted target normalizer.')

            feature_data_to_fit = train_data[..., feature_indices].reshape(T*N, len(feature_indices))
            feature_normalizer.fit(feature_data_to_fit)
            logger.info('Fitted feature normalizer.')

        except Exception as e:
            logger.error(f'Error creating/fitting normalizers: {e}')
            sys.exit(1)
        
        logger.info('Saving normalizers...')

        try:
            save_normalizer(target_normalizer, args.target_normalizer_path)
            save_normalizer(feature_normalizer, args.feature_normalizer_path)
            logger.info(f'Saved target normalizer to {args.target_normalizer_path} and feature normalizer to {args.feature_normalizer_path}')
        except Exception as e:
            logger.error(f'Error saving normalizers: {e}')
            sys.exit(1)

        logger.info('Normalization setup complete.')
        #scale data

    return target_normalizer, feature_normalizer


def main():
    cfg = __init__()

    #model initialization
    model = build_model(cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg.logger.info(f"total params: {total_params:,}")
    #optimizer and scheduler
    optimizer, scheduler = build_optimizer(cfg, model)

    optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=cfg.learning_rate,
    betas=[0.9, 0.999],
    eps=1e-8,
    fused=True,
    )
    
    start_epcoh = 0
    if cfg.resume_model is not None:
        model, optimizer, scheduler, plan = load_checkpoint(cfg.resume_model, model, optimizer, scheduler)
        start_epcoh = plan['epoch']
        cfg.logger.info(f'Resuming training from epoch {start_epcoh}')

    elif cfg.pre_model is not None:
        if hasattr(cfg, 'start_epoch'):
            start_epcoh = cfg.start_epoch
        model, _, _, _ = load_checkpoint(cfg.pre_model, model)
        cfg.logger.info(f'Loaded pre-trained model from {cfg.pre_model}, starting at epoch {start_epcoh}')

    if distributed():
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                broadcast_buffers=False,
                find_unused_parameters=False
        )

    target_normalizer, feature_normalizer = load_normalize(cfg)

    full_data_path = cfg.data_path
    split_ratios = [cfg.train_split_ratio, cfg.valid_split_ratio, 1 - cfg.train_split_ratio - cfg.valid_split_ratio]


    #model attachments
    if cfg.model_name == 'TGCN':
        cfg.adj = torch.from_numpy(np.load(cfg.adj_path)['adj_matrix']).float().to(device)

    #load dataset
    # 1. train
    train_dataset = WindFarmDataset(
        datapath=full_data_path, 
        target_normalizer=target_normalizer,
        feature_normalizer=feature_normalizer,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        flag='train',                
        split_ratio=split_ratios,
        device=device   
    )

    # 2. valid
    valid_dataset = WindFarmDataset(
        datapath=full_data_path, 
        target_normalizer=target_normalizer,
        feature_normalizer=feature_normalizer,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        flag='valid',                  
        split_ratio=split_ratios,
        device=device 
    )

    # 3. test
    test_dataset = WindFarmDataset(
        datapath=full_data_path, 
        target_normalizer=target_normalizer,
        feature_normalizer=feature_normalizer,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        flag='test',                 
        split_ratio=split_ratios,
        device=device 
    )

    # 创建 Dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    start_epoch = 0
    epoch_batch_num = len(train_loader)
    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_batch_num)
    time_counter.reset()


    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    batch_num = len(train_loader.batch_sampler)
    cfg.logger.info('{:#^75}'.format(' Data Information '))
    cfg.logger.info(f'[DATA INFO] ||| [All Batch Num] {batch_num} ||| [Batch Size] {cfg.batch_size} ')
    cfg.logger.info('{:#^75}'.format(' Data Information '))
    loss_epochs = []
    loss_batch = []
    val_loss_epochs = []
    best_val_loss = float('inf')

    criterion = custom_loss(r_rmse=cfg.loss_config.r_rmse, r_mae=cfg.loss_config.r_mae, r_mask1=cfg.loss_config.r_mask1,
                            r_mask2=cfg.loss_config.r_mask2, r_smooth=cfg.loss_config.r_smooth,
                            range_min=cfg.loss_config.range_min, range_max=cfg.loss_config.range_max)
    
    for iepoch in range(start_epcoh, cfg.num_epochs):
        cfg.logger.info('{:#^75}'.format(f' [Train Epoch] {iepoch} '))
        model, optimizer, scheduler, loss_item = train_onepart(cfg, model, train_loader, optimizer, scheduler, iepoch, time_counter, criterion)
        #save_file = os.path.join(cfg.work_dir, cfg.project, 'model', 'epoch_{}.pth'.format(iepoch))
        #cfg.logger.info('epoch_{}.pth'.format(iepoch) + ' Saved')
        valid_loss = valid(cfg, model, valid_loader, iepoch, criterion)

        #plot step loss
        loss_batch.extend(loss_item)
        #draw_loss(loss_batch, iepoch, cfg, 'Batches')
        #batch loss
        loss_mean = np.asarray(loss_item).mean()
        loss_epochs.append(loss_mean)
        val_loss_mean = np.asarray(valid_loss).mean()
        val_loss_epochs.append(val_loss_mean)

        if len(loss_epochs) >= 3:
            draw_loss(loss_epochs, iepoch, cfg, 'Epoches', val_loss_epochs)

        cfg.logger.info(
            '[ Epoch %d ] ||| [lr: %.6f] [Loss: %.4f] |||  MaxMemory %dMB' %
            (iepoch,
             optimizer.param_groups[0]['lr'],
             loss_mean,
             torch.cuda.max_memory_allocated(device) / 1024 ** 2))

        if cfg.scheduler == 'StepLR' or cfg.scheduler == 'CosineLR':
            if iepoch >= cfg.step_size[0]:
                scheduler.step()
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            save_file = os.path.join(cfg.workdir, cfg.project, cfg.model_name, 'best_model', 'best_model_epoch_{}.pth'.format(iepoch))
            save_checkpoint(save_file, model, iepoch, optimizer=optimizer, scheduler=scheduler)
            
            cfg.logger.info(f'Best model saved at epoch {iepoch} with validation loss {best_val_loss:.4f}')

    cfg.logger.info('Training Finished!')
    #valid_loss = valid(cfg, model, valid_loader, iepoch, True)


    #save_file = os.path.join(cfg.work_dir, 'model', 'epoch_{}.pth'.format(iepoch))
    # if iepoch % 5 == 0 and iepoch > 0:
    #save_checkpoint(save_file, model, iepoch, optimizer=optimizer, scheduler=scheduler)


if __name__ == "__main__":
    main()




            


        





        









  




#def train_one_epoch():

#def valid_one_epoch()
    
#def plot()
    


