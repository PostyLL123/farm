import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
import logging
from scipy.stats import norm
import numpy as np

def __init__():
    args = argparse.ArgumentParser(description="Visualize farm data for prediction.")
    args.add_argument('--data_file_path', type=str, default='/home/luoew/stat_data/henan_nanyang/', help='Path to the input CSV data file.')
    args.add_argument('--log_dir', type=str, default='/home/luoew/project/farm_predict/logs/', help='Directory to save the plots.')
    args.add_argument('--work_dir', type=str, default='/home/luoew/project/farm_predict/', help='Working directory for the script.')
    args.add_argument('--plot_dir', type=str, default='/home/luoew/project/farm_predict/plots/', help='Directory to save the plots.')
    args = args.parse_args()
    os.chdir(args.work_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    args.log_file = os.path.join(args.log_dir, 'farm_data_visualization.log')
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

def data_info(file_path, df,args):
    
    args.logger.info(f"turbine{file_path[-6:-4]} Data loaded from {file_path} with shape {df.shape}")
    args.logger.info(f"datanan info:\n{df.isna().sum()}")
    args.logger.info(f"data description:\n{df.describe()}")
    return df

def plot_data(df_list, args):
    """
    绘制风电场数据的各种图表并保存到指定路径。

    参数:
    - df_list: 包含10个风电场数据的DataFrame列表，每个DataFrame应包含'WindSpeed'和'Power'列。
    - args: 包含保存路径的参数对象，需包含属性'plot_dir'。
    """
    # 创建三个子目录
    frequency_dir = os.path.join(args.plot_dir, 'frequency_distribution')
    heatmap_dir = os.path.join(args.plot_dir, 'heatmap')
    normal_dist_dir = os.path.join(args.plot_dir, 'normal_distribution')
    os.makedirs(frequency_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(normal_dist_dir, exist_ok=True)

    # 1. 绘制风速和功率的频次分布图（分别绘制两张图）
    for i, df in enumerate(df_list):
        # 风速频次分布
        plt.figure(figsize=(12, 6))
        plt.hist(df['WindSpeed'], bins=30, color='blue', alpha=0.7)
        plt.xlabel('Wind Speed')
        plt.ylabel('Frequency')
        plt.title(f'Wind Farm {i+1} - Wind Speed Frequency Distribution')
        plt.savefig(os.path.join(frequency_dir, f'wind_farm_{i+1}_WindSpeed_frequency.png'))
        plt.close()

        # 功率频次分布
        plt.figure(figsize=(12, 6))
        plt.hist(df['Power'], bins=30, color='orange', alpha=0.7)
        plt.xlabel('Power')
        plt.ylabel('Frequency')
        plt.title(f'Wind Farm {i+1} - Power Frequency Distribution')
        plt.savefig(os.path.join(frequency_dir, f'wind_farm_{i+1}_Power_frequency.png'))
        plt.close()

    # 2. 绘制风速之间的相关性热力图
    WindSpeed_data = np.column_stack([df['WindSpeed'] for df in df_list])
    correlation_matrix = np.corrcoef(WindSpeed_data, rowvar=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=[f'Farm {i+1}' for i in range(10)], yticklabels=[f'Farm {i+1}' for i in range(10)])
    plt.title('Correlation Heatmap of Wind Speeds')
    plt.savefig(os.path.join(heatmap_dir, 'WindSpeed_correlation_heatmap.png'))
    plt.close()

    # 3. 绘制风速的正态分布图
    plt.figure(figsize=(12, 6))
    x = np.linspace(0, 25, 500)  # 假设风速范围为0到25
    for i, df in enumerate(df_list):
        mean = df['WindSpeed'].mean()
        std = df['WindSpeed'].std()
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, label=f'Farm {i+1}')
    plt.xlabel('Wind Speed')
    plt.ylabel('Density')
    plt.title('Normal Distribution of Wind Speeds')
    plt.legend()
    plt.savefig(os.path.join(normal_dist_dir, 'WindSpeed_normal_distribution.png'))
    plt.close()
    


def main():
    args = __init__()
    data_file_path = args.data_file_path
    df_list = []
    for file_name in os.listdir(data_file_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_file_path, file_name)
            df = pd.read_csv(file_path, sep=',', parse_dates=['time'], index_col='time')
            #data_info(file_path, df, args)
            df_list.append((df))

    #time index对齐
    # 1. 获取所有风机的 "最早" 和 "最晚" 时间戳
    all_min_indices = [df.index.min() for df in df_list]
    all_max_indices = [df.index.max() for df in df_list]

    # 2. 计算交集：
    #    - 交集的开始时间 = 所有风机中 "最晚的" 开始时间
    #    - 交集的结束时间 = 所有风机中 "最早的" 结束时间
    intersection_start = max(all_min_indices)
    intersection_end = min(all_max_indices)

    # (可选) 检查交集是否有效
    if intersection_start >= intersection_end:
        raise ValueError("10个风机的数据时间范围没有重叠，无法创建交集！")

    print(f"--- 成功找到时间交集 ---")
    print(f"公共开始时间: {intersection_start}")
    print(f"公共结束时间: {intersection_end}")

    # 3. 根据交集创建公共索引
    common_index = pd.date_range(start=intersection_start, 
                                end=intersection_end, 
                                freq='7s') # 's' (小写) 是推荐的写法

    # --- END: 新的“交集”索引代码 ---

    # (后续的 for 循环代码保持不变)
    # df_list[i] = df_list[i].reindex(common_index)...
    for i in range(len(df_list)):
        # 确保先处理重复值 (我们上一步的修复)
        df_cleaned = df_list[i].groupby(df_list[i].index).mean()
        
        # 用 "交集" 索引进行 reindex，并插值填补 "内部" 的小缝隙
        df_list[i] = df_cleaned.reindex(common_index, method='nearest')

    plot_data(df_list, args)



if __name__ == "__main__":
    main()
