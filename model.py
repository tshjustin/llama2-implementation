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


class Transformer(nn.Module): 

    def __init__(self, args: ModelArgs) -> None: 
        super().__init__()

        assert args.vocab_size != -1, "Set vocab size"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers # number of decoder layers 
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()

        # create the blocks itself 
        for _ in range(self.n_layers): 
            self.layers.append(EncoderBlock(args))

        
        # batch norm 
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # output layer 
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)  

        # frequency of rotational embeddings 
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, \
            device=self.args.device
        )

        """
        Now in the forward method, just follow the process of information 

        Embeddings --> block layers --> RMS --> final layer --> output
        """

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        This forward method is for inference only!! 

        Sequence length is always 1, since we use KV-cache 

        With KV-Cache, dont have to give all the previously generated tokens,
        just need to give the latest token 

        Hence the intermediate tokens are then cached 

        So the input to the model at anytime step = 1 ==> sequence_length =1 
        """

        # (B, seq_length)
        batch_size, seq_len = tokens.shape()

        assert seq_len == 1, "Only 1 token can be processed at a time"  

        # (B, seq_length) --> # (B, seq_length, dim)
        h = self.tok_embeddings(tokens) # tokens --> dim_size

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_length]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # consecutively apply all the layers 
        for layer in self.layers: 
            h = layer(h, start_pos, freqs_complex)
        
        # final RMS norm 
        h = self.norm(h)

        # final linear layer 
        output = self.output(h).float()

        return output
