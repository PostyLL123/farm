import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist # <-- 我们不再需要这个了
import warnings
import os
import sys

# --- 1. 核心配置：经纬度“地面真相” ---
# (基于您的 image_45154c.png 和 image_4501ad.png)

# 1.1 文件名 -> YY标签 映射 (来自您的 image_4501ad.png)
FILENAME_TO_YY_MAPPING = {
    'combined_01.csv': 'YY1',
    'combined_02.csv': 'YY2',
    'combined_03.csv': 'YY4',
    'combined_04.csv': 'YY5',
    'combined_05.csv': 'YY6',
    'combined_06.csv': 'YY7',
    'combined_07.csv': 'YY8',
    'combined_08.csv': 'YY9',
    'combined_09.csv': 'YY10',
    'combined_10.csv': 'YY12'
}

# 1.2 YY标签 -> 经纬度坐标 映射 (来自您的 image_45154c.png)
# (!! MODIFIED !!)
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

# 1.3 GNN 算法参数 (与您原始脚本一致)
SIGMA_SCALE = 0.1
EPSILON = 0.01

# 1.4 目标文件路径
OUTPUT_DIR = '/home/luoew/project/farm_predict/data/'
NPZ_FILE_NAME = 'windfarm_gnn_unscaled_data.npz'
NPZ_FILE_PATH = os.path.join(OUTPUT_DIR, NPZ_FILE_NAME)


# --- 2. 新增的 Haversine 距离计算函数 ---

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

# --- 3. GNN 矩阵计算函数 (!! MODIFIED !!) ---

def get_adjacency_matrix(farm_order_to_yy, lat_lon_coordinates, sigma_scale, epsilon):
    """
    使用 Haversine 公式计算距离，并生成邻接矩阵 A。
    """
    print(f"--- 步骤 1: 正在计算 *正确* 的GNN邻接矩阵 (使用经纬度) ---")
    num_nodes = len(farm_order_to_yy)
    node_coordinates_latlon = np.zeros((num_nodes, 2))
    
    print("    正在按 Farm 1...10 的顺序构建 (纬度, 经度) 数组:")
    # 按照 'Farm 1', 'Farm 2'... 的顺序构建坐标数组
    for i in range(num_nodes):
        farm_label = f'Farm {i+1}'
        yy_label = farm_order_to_yy[farm_label]
        if yy_label not in lat_lon_coordinates:
            raise KeyError(f"坐标 ' {yy_label}' (来自 {farm_label}) 未在 LAT_LON_COORDINATES 字典中找到。")
        node_coordinates_latlon[i] = lat_lon_coordinates[yy_label]
        print(f"      {farm_label} -> {yy_label} -> (Lat: {lat_lon_coordinates[yy_label][0]}, Lon: {lat_lon_coordinates[yy_label][1]})")

    # (!!) 核心修改：使用 Haversine 计算距离矩阵 (D)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            lat1, lon1 = node_coordinates_latlon[i]
            lat2, lon2 = node_coordinates_latlon[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    print(f"\n    物理距离矩阵 (D) [米] (前5x5):\n {np.round(dist_matrix[:, :], 1)}")

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
    
    print(f"\n    (!! 已修正 !!) GNN 邻接矩阵 (A):\n {np.round(adjacency_matrix, 3)}")
    return adjacency_matrix

# --- 4. 主执行逻辑 (与上个脚本相同) ---

def main_fix():
    print(f"--- 任务开始: 修正 NPZ 文件中的邻接矩阵 ---")
    print(f"    目标文件: {NPZ_FILE_PATH}")

    # 1. 确定 Farm -> YY 映射
    sorted_filenames = sorted(FILENAME_TO_YY_MAPPING.keys())
    num_nodes = len(sorted_filenames)
    
    if num_nodes != 10:
         print(f"错误：MAPPING 字典中的文件数量 ({num_nodes}) 不是 10。", file=sys.stderr)
         return

    farm_order_to_yy = {}
    for i in range(num_nodes):
        filename = sorted_filenames[i]
        yy_label = FILENAME_TO_YY_MAPPING[filename]
        farm_label = f'Farm {i+1}'
        farm_order_to_yy[farm_label] = yy_label
    
    print("--- 确定的 Farm -> YY 映射关系 ---")
    for i in range(num_nodes):
        print(f"    Farm {i+1} (来自 {sorted_filenames[i]}) -> {farm_order_to_yy[f'Farm {i+1}']}")

    # 2. 计算 *新* 的、*正确* 的邻接矩阵
    try:
        # (!!) MODIFIED: 传入经纬度坐标
        new_adj_matrix = get_adjacency_matrix(farm_order_to_yy, LAT_LON_COORDINATES, SIGMA_SCALE, EPSILON)
    except Exception as e:
        print(f"\n!!! 计算邻接矩阵失败: {e}", file=sys.stderr)
        return

    # 3. 加载现有的 NPZ 文件
    try:
        print(f"\n--- 步骤 2: 正在加载现有的 NPZ 文件: {NPZ_FILE_PATH} ---")
        loader = np.load(NPZ_FILE_PATH, allow_pickle=True)
        
        saved_data = {}
        for key in loader.files:
            if key != 'adj_matrix':
                saved_data[key] = loader[key]
                print(f"    保留键: '{key}' (形状: {loader[key].shape})")

        if 'data' not in saved_data:
            print(f"错误：NPZ 文件中缺少 'data' 键。", file=sys.stderr)
            return

    except FileNotFoundError:
        print(f"错误：找不到 NPZ 文件: {NPZ_FILE_PATH}", file=sys.stderr)
        return
    except Exception as e:
        print(f"!!! 加载 NPZ 文件失败: {e}", file=sys.stderr)
        return

    # 4. 保存*新*的 NPZ 文件，替换旧的矩阵
    print(f"\n--- 步骤 3: 正在保存*更新后*的 NPZ 文件 ---")
    TEMP_FILE_PATH = NPZ_FILE_PATH + ".temp"
    
    try:
        np.savez_compressed(
            TEMP_FILE_PATH,
            **saved_data,              # 重新添加所有保留的数据 (data, feature_order)
            adj_matrix=new_adj_matrix  # 添加 *新* 的邻接矩阵
        )
        
        os.replace(TEMP_FILE_PATH, NPZ_FILE_PATH)
        
        print(f"    成功！ {NPZ_FILE_PATH} 已被更新。")
        print("--- 任务完成 ---")

    except Exception as e:
        print(f"!!! 保存 NPZ 文件失败: {e}", file=sys.stderr)
        if os.path.exists(TEMP_FILE_PATH):
            os.remove(TEMP_FILE_PATH)

if __name__ == '__main__':
    main_fix()