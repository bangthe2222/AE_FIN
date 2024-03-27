import numpy as np
import pandas as pd

def generate_random_weights(df_):
    """
    Generate random weights

    Parameters:
    - df_: pd DataFrame

    Returns:
    - weight_df 

    """
    l = [0.01, 0.25, 0.5, 0.75, 1]  # do not use 0, for avoiding zero division
    
    random_weights = np.random.choice(l, size=(df_.shape), replace=True)
    
    random_weights = np.divide(random_weights, random_weights.sum(axis=1)[:,None])
        
    weights_df = pd.DataFrame(random_weights, index=df_.index, columns=df_.columns)
    
    return weights_df

def simulate(assets_returns, assets_weights):
    assets_weights = assets_weights.ffill()
    pf_returns = assets_returns.add(1).mul(assets_weights.shift(1)).sum(axis=1).sub(1)
    pf_returns.iloc[0] = 0  # first day return is 0, because we do not have weights for yesterday
    return pf_returns

def returns_to_equity(returns):
    equity = returns.add(1).cumprod().sub(1)
    return equity
