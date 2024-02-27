# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler

class PreprocessData:
    """
    Preprocess data class
    """
    def __init__(self, df) -> None:
        
        self.company_names = df.columns.to_numpy()[1:]
        self.company_names_out = [name+ " OUT" for name in self.company_names]
        self.data = df.values[:,1:]
        self.data_train = np.array(self.data, dtype=np.float32)
        self.transformer = RobustScaler().fit(self.data_train)
        self.data_norm = self.transformer.transform(self.data_train)

    def get_train_loader(self, batch_size = 64):
        """
        Get train loader

        Parameters:
        - batch_size: batch size of train loader
        
        Returns:
        - train loader
        """
        self.train_loader = DataLoader(self.data_norm, batch_size=batch_size)
        return self.train_loader
    
    def get_inverse_output(self, output):
        """
        Get inverse output data from RobustScaler nomalize

        Parameters:
        - output: output data
        
        Returns:
        - inverse output 
        """
        self.inverse_output = self.transformer.inverse_transform(output.to("cpu").detach().numpy())
        return self.inverse_output
