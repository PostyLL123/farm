import numpy as np
import pandas as pd
import os
import copy
import sys

# --- 路径设置 ---
# 添加项目根目录，以便 utils 等模块能被导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(project_root)

from nowcasting.utils.logger import create_logger
#FILE path settings
workdir = '/home/luoew/model_output/'
# (确保 work_dir 存在)
if not os.path.exists(workdir):
    os.makedirs(workdir, exist_ok=True)
project = 'farm_predict'

#Data settings
data_path = '/home/luoew/project/farm_predict/data/wind_fram_data_henan.npz'
adj_path = '/home/luoew/project/farm_predict/data/adj_matrix.npz'

train_split_ratio = 0.7
valid_split_ratio = 0.2
test_split_ratio = 0.1
target_feature = 'WindSpeed'

#training settings
opt_type = 'adamw'
learning_rate = 0.0001  
weight_decay = 0.0001
scheduler = 'CosineLR' #['CosineLR','StepLR',None]
step_size = [30, 80] # only for StepLR
num_epochs = 100
display_interval = 5
#MODEL SETTING
input_len = 86
output_len = 12
batch_size = 256
input_dim = 11
rand_seed = 1024
hidden_dim = 128
model_name = 'TGCN'

#model
resume_model = None # (保持你原来的逻辑)
pre_model = None # (保持你原来的逻辑)
eval_model = None # (保持你原来的逻辑)

#loss
class loss:
    r_rmse = 0.8
    r_mae = 1.0
    r_mask1 = 1.0
    r_mask2 = 6.0
    r_smooth = 3.5
    range_min = 5
    range_max = 15

loss_config = loss()
#logger
if eval_model is not None:
    # (这似乎是一个错字, 'result_path' 未定义, 改为 'work_dir')
    logger = create_logger(workdir+project+f'/{model_name}', 'test_stage_1') 
else:
    logger = create_logger(workdir+project+f'/{model_name}', 'Train_stage_1')

#Normalization settings
target_normalizer_path = os.path.join(workdir, project, 'normalizer', str(train_split_ratio), 'target_normalizer.json')
feature_normalizer_path = os.path.join(workdir, project, 'normalizer', str(train_split_ratio), 'feature_normalizer.json')
normalizer_method = 'standard' #['minmax','standard']



