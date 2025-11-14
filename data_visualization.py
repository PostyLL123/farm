import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
import logging

def __init__():
    args = argparse.ArgumentParser(description="Visualize farm data for prediction.")
    args.add_argument('--data_file_path', type=str, default='/home/luoew/project/farm_predict/data/henan_nanyang/', help='Path to the input CSV data file.')
    args.add_argument('--log_dir', type=str, default='/home/luoew/project/farm_predict/logs/', help='Directory to save the plots.')
    args.add_argument('--work_dir', type=str, default='/home/luoew/project/farm_predict/', help='Working directory for the script.')
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

def data_process(file_path,args):
    df = pd.read_csv(file_path, sep=',', parse_dates=['time'], index_col='time')
    args.logger.info(f"turbine{file_path[-6:-4]} Data loaded from {file_path} with shape {df.shape}")
    args.logger.info(f"datanan info:\n{df.isna().sum()}")
    args.logger.info(f"data description:\n{df.describe()}")
    return df

def main():
    args = __init__()
    data_file_path = args.data_file_path
    for file_name in os.listdir(data_file_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_file_path, file_name)
            df = data_process(file_path,args)

def plot_data(df, file_name, args):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df)
    plt.title(f"Data Visualization for {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend(df.columns)
    plot_file_path = os.path.join(args.log_dir, f"{file_name}_visualization.png")
    plt.savefig(plot_file_path)
    plt.close()
    args.logger.info(f"Plot saved to {plot_file_path}")

if __name__ == "__main__":
    main()
