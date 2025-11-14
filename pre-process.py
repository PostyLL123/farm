import pandas as pd
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
def __init__():
    
    args = argparse.ArgumentParser(description="Pre-process farm data for prediction.")
    args.add_argument('--data_floder_path', type=str, default='/home/luoew/project/farm_predict/data/henan_nanyang', help='Path to the input CSV data file.')
    #args.add_argument('output_path', type=str, help='Path to save the pre-processed CSV data file.')
    args.add_argument('--work_dir', type=str, default='/home/luoew/project/farm_predict/', help='Working directory for the script.')
    args.add_argument('--select_columns', type=str, nargs='+', default=['信息时间', '风速', '风向', '环境温度', '网侧有功功率'], help='Columns to select from the CSV file.')
    args.add_argument('--column_names', type=str, nargs='+', default=['time', 'WindSpeed', 'WindDirection', 'Temperature', 'Power'], help='Renamed columns for the DataFrame.')
    args = args.parse_args()
    os.chdir(args.work_dir)

    log_dir = os.path.join(args.work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.log_file = os.path.join(log_dir, 'farm_data_process.log')
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

def read_file_to_df(file_path):

    df = pd.read_csv(file_path, sep='\t', encoding='gbk')
    df = df[args.select_columns]
    df.columns = args.column_names
    df['time'] = pd.to_datetime(df['time']).dt.floor('S')
    df.set_index('time', inplace=True)
    #df = df.resample('1S').mean()
    df.loc[df['WindSpeed'] < 0, 'WindSpeed'] = 0
    df.loc[df['WindSpeed'] > 50, 'WindSpeed'] = np.nan

    return df

def read_file_from_folder(data_floder_path):
    global args
    logger = args.logger
    turbine_number = [f"{i:02d}" for i in range(1, 11)]
    
    for number in turbine_number:
        df_list = []
        logger.info(f'Processing turbine 4107850{number}...')
        turbine_floder_path = os.path.join(data_floder_path, f'4107850{number}')
        file_list = os.listdir(turbine_floder_path)
        for file_name in tqdm(file_list, desc=f'Processing turbine 4107850{number} files'):
            file_path = os.path.join(turbine_floder_path, file_name)
            df = read_file_to_df(file_path)
            df_list.append(df)
        combined_df = pd.concat(df_list)
        combined_df.to_csv(data_floder_path + f'/combined_{number}.csv')
        logger.info(f'Finished processing turbine 4107850{number}, saved to combined_{number}.csv')


def main():
    global args
    args = __init__()
    data_floder_path = args.data_floder_path
    #output_path = args.output_path

    read_file_from_folder(data_floder_path)

if __name__ == "__main__":
    main()
