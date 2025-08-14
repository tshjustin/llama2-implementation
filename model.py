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


def precompute_theta_pos_frequencies(
    head_dim: int,
    seq_len: int, 
    device: str, 
    theta: float = 10000.0      
):
    """
    RoPE is the middle ground between absolute and relative embeddings

    1. Key idea is that we want to encode positional information when we perform attention calc

    2. Each token would get rotated in space, by some angle theta, that follows a formla = 1 / 10000^(2i/d), i=index, d=dimension
    
    3. The rotation formula can be derived via the following: 
        a. Each pair of value in the embedding is rotated by the 2D rotation matrix 
        b. Expanding and simplifying gives a concise formula that we can follow --> x_rotated = (x1 x2 ...).(cosmθ1 cosmθ1 cosmθ2 cosmθ2...) + (-x2 x1 ...).(sinmθ1 sinmθ1 sinmθ2 ...)
    """
    assert head_dim % 2 == 0, "Embedding must be even --> 512, 1024..." 

    # theta calculation  
    # shape = (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # now we can build the cos and sim matrics, finding all m and theta, which is all the position of 
    # all the tokens in the sentence --> make use of the sequence length 

    # shape = (seq_len)
    m = torch.arange(seq_len, device=device)

    # multiply with the theta to get mθ --> m_tensor * theta_tensor --> find x_rotated 
    # Outer --> for each element of the first tensor, multiply with each element of the second tensor 
    """
    freqs[0][0] = m[0] × theta[0] = 0 × 1.0 = 0.0
    freqs[0][1] = m[0] × theta[1] = 0 × 0.5 = 0.0
    freqs[1][0] = m[1] × theta[0] = 1 × 1.0 = 1.0
    freqs[1][1] = m[1] × theta[1] = 1 × 0.5 = 0.5
    ...

    freqs = [
    [0.0, 0.0],    # position 0
    [1.0, 0.5],    # position 1  
    [2.0, 1.0]     # position 2
    ]    

    For this case of dim = 512 ==> [m×θ₀, m×θ₁, ..., m×θ₂₅₅] # position m

    also obviously, the shape has became a "2d" in the toy example 

    ==> shape = (seq_len, head_dim / 2)
    """
    freqs = torch.outer(m, theta).float()
    
    # compute compelx num in polars form c = R * exp(1 * m * theta), R = 1
    # the reason to put to polars form is because we want to follow a series of transformation to apply these positonal info to the embeddings 
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(
    x: torch.Tensor, 
    freqs_complex: torch.Tensor, 
    device: str
): 
    # 1. Rehape (B, seq_len, H, Head_Dim) -> (B, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 2. (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # 3. element wise operation (B, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim / 2)
    # (B , seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # 4. simply and put to the form we want at the start 
    # (B , seq_len, h, head_dim / 2) -> (B, seq_len, h, head_dim / 2, 2) --> Tensor of 2D
    x_out = torch.view_as_real(x_rotated)

    # 5. reshape  (B, seq_len, h, head_dim / 2, 2) ->  (B, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)



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
