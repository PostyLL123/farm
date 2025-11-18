import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
import logging
from scipy.stats import norm
import numpy as np
import json  # (我们将用它来保存 scalers, 但在训练脚本中)
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings

# --- 1. 初始化 (来自您的代码, 无变化) ---
def __init__():
    args = argparse.ArgumentParser(description="Visualize farm data for prediction.")
    args.add_argument('--data_file_path', type=str, default='/home/luoew/stat_data/henan_nanyang/', help='Path to the input CSV data file.')
    args.add_argument('--log_dir', type=str, default='/home/luoew/project/farm_predict/logs/', help='Directory to save the plots.')
    args.add_argument('--work_dir', type=str, default='/home/luoew/project/farm_predict/', help='Working directory for the script.')
    args.add_argument('--output_dir', type=str, default='/home/luoew/project/farm_predict/data/', help='Directory to save the featured data.')
    args = args.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) # 确保输出目录存在
    os.chdir(args.work_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    args.log_file = os.path.join(args.log_dir, 'feature_project.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    args.logger = logging.getLogger(__name__)
    return args

# --- 2. 新增的功能函数 ---

# --- 步骤 3: 特征工程 (与之前相同) ---
def create_all_features(df_list, logger, window_size='10min'):
    logger.info(f"--- 步骤 3: 正在创建特征 (窗口: {window_size}) ---")
    processed_dfs = []
    
    for i, df in enumerate(tqdm(df_list, desc="    Creating Features", unit="file", ncols=100)):
        try:
            core_cols = ['WindSpeed', 'WindDirection']
            df_core = df[core_cols].copy()
            new_df = pd.DataFrame(index=df_core.index) 
            new_df['WindSpeed'] = df_core['WindSpeed']
            wind_dir_rad = np.deg2rad(df_core['WindDirection'])
            new_df['Wind_U'] = -df_core['WindSpeed'] * np.sin(wind_dir_rad)
            new_df['Wind_V'] = -df_core['WindSpeed'] * np.cos(wind_dir_rad)
            new_df['hour_sin'] = np.sin(2 * np.pi * new_df.index.hour / 24.0)
            new_df['hour_cos'] = np.cos(2 * np.pi * new_df.index.hour / 24.0)
            new_df['month_sin'] = np.sin(2 * np.pi * new_df.index.month / 12.0)
            new_df['month_cos'] = np.cos(2 * np.pi * new_df.index.month / 12.0)
            ws_rolling_mean = new_df['WindSpeed'].rolling(window=window_size, min_periods=1).mean()
            u_rolling_mean = new_df['Wind_U'].rolling(window=window_size, min_periods=1).mean()
            v_rolling_mean = new_df['Wind_V'].rolling(window=window_size, min_periods=1).mean()
            ws_rolling_std = new_df['WindSpeed'].rolling(window=window_size, min_periods=1).std()
            new_df['TurbulenceIntensity'] = ws_rolling_std / (ws_rolling_mean + 1e-6)
            new_df['WS_rolling_mean'] = ws_rolling_mean
            new_df['U_rolling_mean'] = u_rolling_mean
            new_df['V_rolling_mean'] = v_rolling_mean
            new_df = new_df.fillna(method='bfill')
            new_df = new_df.fillna(0)
            processed_dfs.append(new_df)
        except KeyError as e:
            logger.error(f"    处理文件 {i} 时出错：缺少列 {e}。跳过此文件。")
            
    logger.info("    特征创建完毕。")
    return processed_dfs

# --- 步骤 4：计算邻接矩阵 (与之前相同) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个 (纬度, 经度) 点之间的地球表面距离 (单位：米)。
    """
    R = 6371000  # 地球半径，单位：米

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def get_adjacency_matrix(farm_order_to_yy, lat_lon_coordinates, logger, sigma_scale=0.1, epsilon=0.01):
    """
    使用 Haversine 公式计算距离，并生成邻接矩阵 A。
    """
    logger.info(f"--- 步骤 1: 正在计算 *正确* 的GNN邻接矩阵 (使用经纬度) ---")
    num_nodes = len(farm_order_to_yy)
    node_coordinates_latlon = np.zeros((num_nodes, 2))
    
    logger.info("    正在按 Farm 1...10 的顺序构建 (纬度, 经度) 数组:")
    # 按照 'Farm 1', 'Farm 2'... 的顺序构建坐标数组
    for i in range(num_nodes):
        farm_label = f'Farm {i+1}'
        yy_label = farm_order_to_yy[farm_label]
        if yy_label not in lat_lon_coordinates:
            raise KeyError(f"坐标 ' {yy_label}' (来自 {farm_label}) 未在 LAT_LON_COORDINATES 字典中找到。")
        node_coordinates_latlon[i] = lat_lon_coordinates[yy_label]
        logger.info(f"      {farm_label} -> {yy_label} -> (Lat: {lat_lon_coordinates[yy_label][0]}, Lon: {lat_lon_coordinates[yy_label][1]})")

    # (!!) 核心修改：使用 Haversine 计算距离矩阵 (D)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            lat1, lon1 = node_coordinates_latlon[i]
            lat2, lon2 = node_coordinates_latlon[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    logger.info(f"\n    物理距离矩阵 (D) [米] (前5x5):\n {np.round(dist_matrix[:, :], 1)}")

    # 计算邻接矩阵 (A) - 高斯核
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma_sq = (dist_matrix[dist_matrix > 0].mean()**2) * sigma_scale
        if np.isnan(sigma_sq) or sigma_sq == 0: sigma_sq = 1.0
        dist_sq = dist_matrix**2
        adjacency_matrix = np.exp(-dist_sq / sigma_sq)
    
    # 稀疏化 (剪枝)
    adjacency_matrix[adjacency_matrix < epsilon] = 0
    # (!!) 修正：确保对角线为1
    np.fill_diagonal(adjacency_matrix, 1.0)

    return adjacency_matrix
# --- 步骤 5：堆叠并保存 (!! MODIFIED !!) ---
def stack_and_save_to_npz(processed_dfs, adj_matrix, output_dir, logger):
    """
    将(未归一化的)特征堆叠为3D数组，并与邻接矩阵一起保存到 .npz 文件。
    """
    logger.info(f"--- 步骤 5: 正在堆叠并保存数据 ---")
    
    # 1. 循环，将 DF 转换为 Numpy 数组
    final_data_list = []
    for df in tqdm(processed_dfs, desc="    Converting to Numpy", unit="file", ncols=100):
        final_data_list.append(df.to_numpy())

    # 2. 堆叠为 (T, N, F) 3D 数组
    logger.info("    正在将数据堆叠为 3D 数组...")
    # (T = 时间步, N = 节点(10), F = 特征(11))
    final_3d_array = np.stack(final_data_list, axis=1)
    
    # 3. 获取特征顺序 (在训练脚本中需要这个)
    feature_order = list(processed_dfs[0].columns)
    logger.info(f"    最终的 3D 数据形状: {final_3d_array.shape}")
    logger.info(f"    特征顺序: {feature_order}")

    # 4. 保存为单一的 .npz 文件
    output_path_npz = os.path.join(output_dir, 'windfarm_gnn_unscaled_data.npz')
    logger.info(f"    正在压缩和保存 .npz 文件 (这可能需要几分钟)...")
    
    # (!! MODIFIED !!) 我们将特征顺序也保存到 .npz 文件中
    np.savez_compressed(
        output_path_npz,
        data=final_3d_array,              # 键 'data' 对应 3D 时序数据
        adj_matrix=adj_matrix,      # 键 'adj_matrix' 对应邻接矩阵
        feature_order=np.array(feature_order) # 键 'feature_order' 对应特征列表
    )
    logger.info(f"    (未归一化)数据已保存到: {output_path_npz}")
    
# --- 3. 您的 MAIN 函数 (已修改) ---

def main():
    args = __init__()
    args.logger.info("--- 任务开始: GNN 原始数据预处理 ---")
    
    # --- 1. 项目配置 (您必须在这里修改) ---
    FILENAME_TO_YY_MAPPING = {
        'combined_01.csv': 'YY1', 'combined_02.csv': 'YY2',
        'combined_03.csv': 'YY4', 'combined_04.csv': 'YY5',
        'combined_05.csv': 'YY6', 'combined_06.csv': 'YY7',
        'combined_07.csv': 'YY8', 'combined_08.csv': 'YY9',
        'combined_09.csv': 'YY10', 'combined_10.csv': 'YY12'
    }
    LAT_LON_COORDINATES = {
        'YY1':  (35.00698213, 114.09582558), # (纬度, 经度)
        'YY2':  (35.00622773, 114.10449460),
        'YY3':  (35.00704826, 114.11703941), # 备选
        'YY4':  (35.00200424, 114.14529942),
        'YY5':  (34.99946417, 114.16235506),
        'YY6':  (35.00678586, 114.19885913),
        'YY7':  (35.00486116, 114.21231153),
        'YY8':  (35.00562687, 114.22845357),
        'YY9':  (34.98221293, 114.22826586),
        'YY10': (34.98636518, 114.21618257),
        'YY11': (34.97992119, 114.23437093), # 备选
        'YY12': (34.96742042, 114.23622048)
    }
    WINDOW_SIZE_STR = '10min'
    MASTER_NODE_INDEX = 0
    
    # --- 步骤 1: 加载数据 ---
    args.logger.info("--- 步骤 1: 正在加载和清理数据 ---")
    data_file_path = args.data_file_path
    df_list = []
    load_order_filenames = []
    
    file_list = sorted([f for f in os.listdir(data_file_path) if f.endswith('.csv')])
    if not file_list:
        args.logger.error(f"在 {data_file_path} 中没有找到 .csv 文件")
        return

    for file_name in tqdm(file_list, desc="    Loading CSVs", unit="file", ncols=100):
        if file_name not in FILENAME_TO_YY_MAPPING:
            args.logger.warning(f"    跳过文件 {file_name}，因为它未在 MAPPING 中定义。")
            continue
            
        file_path = os.path.join(data_file_path, file_name)
        df = pd.read_csv(file_path, sep=',', parse_dates=['time'], index_col='time')
        
        if df.index.has_duplicates:
            df = df.groupby(df.index).mean()
            
        df_list.append((df))
        load_order_filenames.append(file_name)
        
    num_nodes = len(df_list)
    args.logger.info(f"成功加载 {num_nodes} 个风机数据。")

    # --- 步骤 2: 时间对齐 (您的代码) ---
    args.logger.info(f"--- 步骤 2: 正在对齐时间戳")
    
    try:
        intersection_start = max(df.index.min() for df in df_list)
        intersection_end = min(df.index.max() for df in df_list)
        if intersection_start >= intersection_end: raise ValueError("数据时间范围没有重叠！")
        args.logger.info(f"    找到公共时间范围: {intersection_start} 到 {intersection_end}")

        master_index = pd.date_range(start=intersection_start, 
                                end=intersection_end, 
                                freq='7s')
        args.logger.info(f"    创建 '时间轴' 成功，包含 {len(master_index)} 个时间点。")

        aligned_df_list = []
        for i in tqdm(range(num_nodes), desc="    Aligning Data", unit="file", ncols=100):
            df_cleaned = df_list[i].groupby(df_list[i].index).mean()
            df_aligned = df_cleaned.reindex(master_index, method='nearest')
            aligned_df_list.append(df_aligned)
        args.logger.info("    所有风机已成功对齐到 '主时间轴'。")

    except Exception as e:
        args.logger.error(f"!!! 时间对齐失败: {e}")
        return

    # --- 步骤 3: 特征工程 ---
    processed_dfs = create_all_features(aligned_df_list, args.logger, WINDOW_SIZE_STR)

    # --- 步骤 4: 计算邻接矩阵 ---
    try:
        farm_order_to_yy = {
            f'Farm {i+1}': FILENAME_TO_YY_MAPPING[load_order_filenames[i]]
            for i in range(num_nodes)
        }
        adjacency_matrix = get_adjacency_matrix(farm_order_to_yy, LAT_LON_COORDINATES, args.logger)
    except Exception as e:
        args.logger.error(f"!!! GNN 邻接矩阵创建失败: {e}")
        return

    # --- 步骤 5: 堆叠并保存 (!! MODIFIED !!) ---
    try:
        # (已移除归一化)
        stack_and_save_to_npz(processed_dfs, adjacency_matrix, args.output_dir, args.logger)
    except Exception as e:
        args.logger.error(f"!!! 数据保存失败: {e}")
        return
        
    args.logger.info("--- 任务成功完成! ---")


if __name__ == '__main__':
    main()