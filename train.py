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
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
print(f"Using GPU: {use_gpu}")

project_rot = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_rot not in sys.path:
    sys.path.append(project_rot)

from utils.commn import save_config, __mkdir__
from utils.normalizer import DataNormalizer, create_normalizer, save_normalizer, load_normalizer

def __init__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='TGCN')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--project', type=str, default='farm predict', help='the farm turbines windspeed and direciton predict')

    args = parser.parse_args()

    config_file = 'main.config.' + args.config
    try:
        xconfig = importlib.import_module(config_file)
    except ImportError:
        print(f'makesure {config_file} exsited')
        sys.exit(1)

    #config_save_path = os.path.join(xconfig.workdir, args.config)
    save_config(config_file, xconfig.workdir)

    xconfig.logger.info(f"[WORK DIR]: {xconfig.workdir}")
    xconfig.logger.info(f'[Configs]: {config_file}')
    
    torch.manual_seed(xconfig.rand_seed)
    np.random.seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)

    if use_gpu:
        torch.cuda.manual_seed_all(xconfig.rand_seed)

    return xconfig

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
    args = __init__()
    target_normalizer, feature_normalizer = load_normalize(args)

if __name__ == "__main__":
    main()




            


        





        









  




#def train_one_epoch():

#def valid_one_epoch()
    
#def plot()
    


