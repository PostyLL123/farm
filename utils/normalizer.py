import numpy as np
import pandas as pd
import json
import os
from typing import Union, List, Tuple, Dict

class DataNormalizer:
    """data normalizer generator, support multiple normalization methods and different shapes of data"""
    def __init__(self, method, feature_range=(-1, 1)):
        """
        initialize normalizer
        
        Args:
            method: normalization method, optional 'standard', 'minmax', 'robust'
            feature_range: target range for minmax method
        """
        self.method = method
        self.feature_range = feature_range
        self.params = {}
        self._is_fitted = False

    def fit(self, data):
        """calculate normalization parameters
        
        params:
            data: input data, can be numpy array or pandas DataFrame
            columns: list of column names to normalize (only for DataFrame)

        """

        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns.tolist()
            values = data[columns].values
        else:
            values = data
        
        #makesure the data is 1D array at least
        if values.ndim == 0:
            values = values.reshape(-1, 1)       
        elif values.ndim == 1:
            values = values.reshape(-1, 1)

        #calculate normalization parameters
        if self.method == 'standard':
            self.params['mean'] = np.mean(values, axis=0)
            self.params['std'] = np.std(values, axis=0)
            #avoid division by zero
            self.params['std'][self.params['std'] == 0] = 1.0

        elif self.method == 'minmax':
            pass
        elif self.method == 'robust':
            pass

        self._is_fitted = True

    def transform(self, data):
        """apply normalization transformation
        
        params:
            data: input data
            columns: list of column names to normalize (only for DataFrame)
            
        
        return:
            normalized data
        """

        if not self._is_fitted:
            raise ValueError('Please fit the normalizer first')
        
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            if columns is None:
                columns = data.columns.tolist()
            values = data[columns].values
        else:
            values = data.copy()

        if values.dim == 1 and len(self.params.get('mean', [0])) == 1:
            values = values.reshape(-1, 1)

        if self.method == 'standard':
            values = (values - self.params['mean']) / self.params['std']

        elif self.method == 'minmax':
            pass

        if is_dataframe:
            data_copy = data.copy()
            data_copy[columns] = values
            return data_copy
        return values
    
    def fit_transform(self, data):
        """fit and transform in one step"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """inverse transform normalized data
        
        params:
            data: normalized data
            columns: list of column names to inverse transform (only for DataFrame)
            
        
        return:
            data in original scale
        """

        if not self._is_fitted:
            raise ValueError('Please fit the normalizer first')
        
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            if columns is None:
                columns = data.columns.tolist()
            values = data[columns].values
        else:
            values = data.copy()

        if values.dim == 1 and len(self.params.get('mean', [0])) == 1:
            values = values.reshape(-1, 1)

        if self.method == 'standard':
            values = values * self.params['std'] + self.params['mean']

        elif self.method == 'minmax':
            pass

        if is_dataframe:
            data_copy = data.copy()
            data_copy[columns] = values
            return data_copy
        return values
    

    def save(self, filepath):
        """save normalization parameters to a json file"""

        json_safe_params = {}
        for key, values in self.params.items():
            if isinstance(values, np.ndarray):
                json_safe_params[key] = values.tolist()
            else:
                json_safe_params[key] = values

        save_dict = {
            'method': self.method,
            'feature_range': self.feature_range,
            'params': json_safe_params,
            'is_fitted': self._is_fitted
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(save_dict, f)

        print(f'Normalizer parameters saved to {filepath}')


    @classmethod
    def load(cls, filepath):
        """load normalization parameters from a json file"""
    
        with open(filepath, 'r') as f:
            save_dict = json.load(f)

        normalizer = cls(method=save_dict['method'], feature_range=tuple(save_dict['feature_range']))

        loaded_params = save_dict['params']

        for key, values in loaded_params.items():
            if isinstance(values, list):
                normalizer.params[key] = np.array(values)
            else:
                normalizer.params[key] = values

        normalizer._is_fitted = save_dict['is_fitted']
        print('Normalizer parameters loaded from {}'.format(filepath))
        return normalizer
    
def create_normalizer(data: Union[np.ndarray, pd.DataFrame], 
                      method: str = 'standard', 
                      feature_range: Tuple[float, float] = (-1, 1), 
                      columns: List[str] = None) -> DataNormalizer:
    """create and fit a DataNormalizer
    
    params:
        data: input data
        method: normalization method
        feature_range: target range for minmax method
        columns: list of column names to normalize (only for DataFrame)
        
    return:
        fitted DataNormalizer
    """
    normalizer = DataNormalizer(method=method, feature_range=feature_range)
    normalized_data = normalizer.fit_transform(data)
    return normalizer, normalized_data

def save_normalizer(normalizer: DataNormalizer, filepath: str):
    """save DataNormalizer to file"""
    normalizer.save(filepath)

def load_normalizer(filepath: str) -> DataNormalizer:
    """load DataNormalizer from file"""
    return DataNormalizer.load(filepath)