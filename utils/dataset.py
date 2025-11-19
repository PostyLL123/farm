import torch
from torch.utils.data import Dataset
import numpy as np
import os

from farm_predict.utils.normalizer import DataNormalizer

class WindFarmDataset(Dataset):
    def __init__(self,
                 datapath: str,
                 feature_normalizer: DataNormalizer,
                 target_normalizer: DataNormalizer,
                 input_len: int,
                 output_len: int,
                 flag: str = 'train',
                 split_ratio: list = [0.7, 0.2, 0.1],
                 target_feature: str = 'WindSpeed',
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.device = device

        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Data file not found: {datapath}")
        
        loader = np.load(datapath, allow_pickle=True)
        raw_data_full = loader['data']
        feature_order = loader['feature_order']

        total_len = raw_data_full.shape[0]
        len_train = int(total_len * split_ratio[0])
        len_valid = int(total_len * split_ratio[1])

        if flag == 'train':
            raw_data_slice = raw_data_full[:len_train]
        elif flag == 'valid':
            raw_data_slice = raw_data_full[len_train:len_train + len_valid]
        else:  # 'test'
            raw_data_slice = raw_data_full[len_train + len_valid:]

        print(f"{flag} data shape: {raw_data_slice.shape}")



        try:
            self.target_idx = np.where(feature_order == target_feature)[0][0]

        except IndexError:
            raise ValueError(f"Target feature '{target_feature}' not found in feature order.")
        
        self.feature_idx = [i for i in range(len(feature_order)) if i != self.target_idx]

        self.data_scaled = self._apply_normalization(raw_data_slice, target_normalizer, feature_normalizer)

        self.n_samples = self.data_scaled.shape[0]-(self.input_len + self.output_len)+1

        self.data_tensor = torch.FloatTensor(self.data_scaled).to(device)

    def _apply_normalization(self, data, t_scaler, f_scaler):
        T, N, F = data.shape
        scaled_data = np.zeros_like(data, dtype=np.float32)

        target_vals = scaled_data[..., self.target_idx].reshape(-1, 1)
        scaled_target = t_scaler.transform(target_vals)
        scaled_data[..., self.target_idx] = scaled_target.reshape(T, N)


        feature_vals = data[..., self.feature_idx].reshape(-1, len(self.feature_idx))
        scaled_features = f_scaler.transform(feature_vals)
        for i, feat_idx in enumerate(self.feature_idx):
                scaled_data[..., feat_idx] = scaled_features[:, i].reshape(T, N)

        return scaled_data
    

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_in = idx
        end_in = idx + self.input_len
        start_out = end_in
        end_out = start_out + self.output_len
        
        X = self.data_tensor[start_in:end_in, :, :]
        Y = self.data_tensor[start_out:end_out, :, self.target_idx].unsqueeze(-1)
        
        return X, Y        


