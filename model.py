import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math 

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs: 
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # query heads 
    n_kv_heads: Optional[int] = None # groups of KV heads 

    vocab_size: int = -1 # set when we load in tokenzier 

    # FFN has more parameters to make up for the lesser K and V heads  
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None 

    # batch normalization parameters 
    norm_eps = 1e-5 
    
    # training params 
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    # misc
    device: str = None 


cl