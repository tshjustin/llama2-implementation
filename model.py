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

def repeat_kv(x: torch.Tensor, n_rep: int): 
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module): 
    """
    Layer normalization 

    proposed that recentering dont contribute much, but instead rescaling 

    hence find a statistic that is independent of the mean ==> RMS statistic instead 
    """
    def __init__(self, dim: int, eps: float = 1e-6): 
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim)) # gamma param 

    def _norm(self, x: torch.Tensor): 
        # (B, seq_len, dim) * (B, seq_len, 1) = (B, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor): 
        # (dim) * (b, seq_len, dim) = (b, seq_len, dim)
        # gamma * ai_hat
        return self.weight * self._norm(x.float().type_as(x))

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        swish allows for smooth negatives , allowing for better information flows 
        
        """
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim) --> linear transforation + swish activation -> (1, 1, 512) -> (1, 1, 1408) --> Since w1 and 3 works sequentially, must compenstate for the increase so 2/3 multiply 
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim) --> linear transformation + no activation -> (1, 1, 512) -> (1, 1, 1408)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim) --> SwiGLU = multiply them together (the "gating") --> (1, 1, 1408) * (1, 1, 1408) -> (1, 1, 1408)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim) --> project back to orignal size --> (1, 1, 1408) -> (1, 1, 512)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs): 
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim 
        self.head_dim = args.dim // args.n_heads 

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalize before the self attention 
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # normalizr before the FFN block 
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor): 
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        # apply attention on the normalized input embeddings 
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)

        # output after the skip connection 
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out 
    
class SelfAttention(nn.Module):
    """
    1. split input embeddings to different heads 

    2. apply the self-attention, where the Q interacts with K

    3. self attention since all the Q K V comes from the same place --> not cross attention 

    4. In the inferecne case --> still the same --> for the nth word, would use information from 1...n-1th word via self attention 

    5. KV cache comes into play here so we dont have to calculate K, V again AND dont have to keep the previous tokens that were generated
    
    6. But MHA + KV-Cache --> alot of reads to the cache --> so reduce the amount of reads by lowering K V matrices 

    7. So use GQA, N groups, k KV-groups that are shared amongst groups 
    """ 
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Compared to the initlaly llama, this one dont have any parallization 

        # KV - Heads 
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # Q Heads 
        self.n_heads_q = self.n_heads 
        
        # The ratio of number of Q heads to KV-Heads --> how many times k,v repeated / query 
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # part of the embeddings that are "visualized" by each head 
        self.head_dim = args.dim // args.n_heads 

        # K , Q , V matrics 
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads, args.n_heads * self.head_dim, bias=False)

        # Cache --> one for key and one for values 
        """
        During Inference 

        Given the input length / prompt 

        1. prompt gets multiplied with K Q V matrics 

        2. results in k q v vectors that are shareded to different heads 

        3. since we are use GQA, K V < Q 

        for each token in the input sequence:
            they have a corresponding q k v vector 
            
            each k q v vector are then sharded to a smaller head / different view of the embedding 
        
        cache_k = [
            # Batch 0 - batch_size=1, seq_len=4, n_kv_heads=2, head_dim=3
            [
                # Position 0: [[k_head0], [k_head1]]
                [[1.2, 0.3, -0.1], [0.8, -0.2, 0.5]],

                # Position 1: 
                [[0.9, 0.1, 0.7], [0.3, 0.4, -0.3]],

                # Position 2, 3...
                [[...], [...]],
                [[...], [...]]
            ]

        Use of the KV-Cache 

        Consider the autoregressive style, and we have already generated "Cat Sat" 

        Without KV_cache 
        -----------------
        all_embeddings = embeddings("Cat Sat")

        all_k = all_embeddings @ Wk
        all_v = all_embeddings @ Wv 

        attention --> generates "on" 

        # now for the next word 
        all_embeddings = embeddings("Cat Sat on")

        all_k = all_embeddings @ Wk 
        all_v = all_embeddings @ Wv 

        # Notice that we recalculate the Key and Value vector again for "Cat Sat' --> Cache them instead 

        With KV_Cache 
        ---------------        
        all_embeddings = embeddings("Cat Sat")

        all_k = all_embeddings @ Wk
        all_v = all_embeddings @ Wv 

        store_cache(all_k, all_v)

        attention --> generates "on" 

        # now for the next word 
        all_embeddings = embeddings("Cat Sat on")

        get_embeddings([:n-1])

        all_k = embedding("on") @ Wk 
        all_v = embedding("on") @ Wv 
        ...

        """
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape() # (B, 1, dim)

        # multiply with the query , key and value matrix 
        # (B, 1, dim) --> (B, 1, HQ * head_dim)
        xq = self.wq(x)

        # (B, 1, dim) --> (B, 1, HKV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, HQ * head_dim) --> (B, 1, HQ * head_dim)
        # split to different attention heads --> since self_n_kv_heads != n_heads_q, would be different number of resulting vectors
        # [q1, q2, q3, ..., q512]  -> [q1, q2, ..., q64], [q65, q66, ..., q128]...
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # now add the rotary embeddings --> dont affect the shape at all 
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # KV - Cache 
        # Replace the entry in the cache for this token 
        """
        This operation have to deal with how the cache looks like and how we store in the information 

        cache_k = [
            # Batch size = 1 --> Since inference 
            [
                # Position 0: "The" 
                [[k_head0_dims], [k_head1_dims]],
                
                # Position 1: "cat"
                [[k_head0_dims], [k_head1_dims]], 
                
                # Position 2: "sat" 
                [[k_head0_dims], [k_head1_dims]],
                
                # Position 3: empty (zeros)
                [[0, 0, 0, ...], [0, 0, 0, ...]],
                
                # Position 4-9: empty (zeros)
                [[0, 0, 0, ...], [0, 0, 0, ...]],
            ]
        ]

        cache_k[:batch_size, start_pos:start_pos+seq_len]
        
        a. Takes the first batch (1 in this case since inferece)

        b. From this batch size, take the K_head embeddings from start_pos:seq_len, where the range typically is the length of the generated sequence 
        """

        # Update cache 
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # Retrieve all the cached keys and values so far 
        # (B, seq_len_kv, head_kv, head_dim)
        # Retrieve cache 
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of the queries 
        """
        Since number of heads of key and values != query heads 

        We need to account for this in the calculation 
        
        1. Copy the single k/v head into multiple heads into a MHA architecture 

        # This is also known as a 2KV Head, lol even though V is misleading here 
        keys = [
            [
                # Position 0: "The"
                [k_head0, k_head1],

                # Position 1: "cat" 
                [k_head0, k_head1],

                # Position 2: "sat"
                [k_head0, k_head1]
            ]
        ]

        # To make it the KV head work with the Query heads, multiply the KV heads 

        # After repeat_kv(keys, n_rep=4) - 8 KV heads:
        keys = [
            [
                # Position 0: "The"
                [k_head0, k_head0, k_head0, k_head0, k_head1, k_head1, k_head1, k_head1], --> note that its repeated, saves parameter counts on K/V 

                # Position 1: "cat"
                [k_head0, k_head0, k_head0, k_head0, k_head1, k_head1, k_head1, k_head1], 

                # Position 2: "sat"
                [k_head0, k_head0, k_head0, k_head0, k_head1, k_head1, k_head1, k_head1]
            ]
        ]
        """

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Now move the head dimension before sequence, since each head just reads a part of the embedding 
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)

        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)

        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        # Concatenation 
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)
    
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
            self.args.dim // self.args.n_heads, 
            self.args.max_seq_len * 2, 
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
