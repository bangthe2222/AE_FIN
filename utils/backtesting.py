# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader

class WalkForwardBackTesting:
    """
    Walk Forward BackTesting Class
    
    """
    def __init__(self, data, num_run, percent = 2) -> None:
        self.data_norm = data
        self.num_run = num_run
        self.len_data = data.shape[0]
        self.forward_data = int(percent*self.len_data/100)
    def get_data_loader(self, batch = 16, train_percent = 0.7):
        """
        Create Walk Forward dataloader

        Parameters:
        - batch: batch size
        - train_percent: percent of train dataset 

        Returns:
        - list of dataloader
        
        """
        list_loader = []
        for i in range(self.num_run):
            data_split = self.data_norm[i*self.forward_data: self.len_data - int(self.num_run - i - 1)*self.forward_data,:]
            len_split = data_split.shape[0]
            train_num = int(train_percent*len_split)
            # test_num = len_split - train_num
            train_split = data_split[:train_num,:]
            test_split = data_split[train_num:,:]
            list_loader.append([DataLoader(train_split, batch), DataLoader(test_split, batch)])
        return list_loader

class kfoldCrossValidationBackTesting:
    """
    k-fold Cross Validation BackTesting Class
    
    """
    def __init__(self, data, num_run) -> None:
        self.data = data
        self.num_run = num_run
        self.len_data = len(data)
        
    def get_data_loader(self, batch=16):
        """
        k-fold Cross Validation dataloader

        Parameters:
        - batch: batch size

        Returns:
        - list of dataloader
        
        """
        list_loader = []
        fold_size = self.len_data // self.num_run
        
        for i in range(self.num_run):
            # Calculate start and end indices of the test fold
            test_start = i * fold_size
            test_end = (i+1) * fold_size if i < self.num_run - 1 else self.len_data
            
            # Split data into training and testing parts
            test_part = self.data[test_start:test_end]
            train_part = np.concatenate([self.data[:test_start], self.data[test_end:]], axis=0)
            
            list_loader.append([DataLoader(train_part, batch), DataLoader(test_part, batch)])
        
        return list_loader

