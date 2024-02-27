# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt 
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from model.AE_FIN import loss_function
        
def normalize_histogram(hist):
    """
    Normalize a histogram so that it represents a probability distribution.
    
    Parameters:
    - hist: NumPy array representing the histogram counts.
    
    Returns:
    - Normalized histogram as a NumPy array.
    """
    total = np.sum(hist)
    if total == 0:
        raise ValueError("Histogram sum is zero; cannot normalize.")
    return hist / total

def total_variation_distance(distribution1, distribution2):
    """
    Calculate the Total Variation Distance (TVD) between two probability distributions.
    
    Parameters:
    - distribution1: NumPy array representing the first probability distribution.
    - distribution2: NumPy array representing the second probability distribution.
    
    Returns:
    - The TVD between the two distributions.
    """
    # Check if the distributions are normalized
    if not np.isclose(np.sum(distribution1), 1) or not np.isclose(np.sum(distribution2), 1):
        raise ValueError("Distributions must be normalized")
    
    # Calculate the TVD
    tvd = 0.5 * np.sum(np.abs(distribution1 - distribution2))
    return tvd

def skew_to_alpha(skew):
    """
    Convert a skew to alpha parameter needed by scipy_stats.skewnorm(..).

    Parameters
    ----------
    skew: float
        Must be between [-0.999, 0.999] for avoiding complex numbers.

    Returns
    -------
    float
    """
    d = (np.pi / 2 * ((abs(skew) ** (2 / 3)) / (abs(skew) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3)))) ** 0.5
    a = (d / ((1 - d ** 2) ** .5))
    return a * np.sign(skew)


def moments(returns):
    """
    Calculate the four moments: mean, std, skew, kurtosis.

    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame

    Returns
    -------
    pd.Series, pd.DataFrame
    """
    if type(returns) != pd.DataFrame:
        return pd.Series({'mean': np.mean(returns),
                          'std': np.std(returns, ddof=1),
                          'skew': scipy_stats.skew(returns),
                          'kurt': scipy_stats.kurtosis(returns, fisher=False)})
    else:
        return returns.apply(moments, axis=1)
