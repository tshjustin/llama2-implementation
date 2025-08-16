import torch 

# m = torch.arange(10) # tensor obj of [0:10]

# print(m)

batch_size = 2
seq_len = 3  
num_heads = 4
head_dim = 8  

q = torch.randn(2, 4, 3, 8)  # (batch, heads, seq_len, head_dim)
k = torch.randn(2, 4, 3, 8)

# position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])  # (batch, seq_len)


position_ids = [[0, 1, 2], [0, 1, 2]]  # (2, 3)
inv_freq = [1.0, 0.1, 0.01, 0.001]     # (4,)

# Expand for batch processing:
"""
Notice that the shapes are not comptabile to perform matrix multiplcation 

1. We want every combination of, for each batch, for each frequency, for each position 


Given frequency = f1 f2 f3 ... 
Given position = p1 p2 p3 ... 

we want f1p1 f1p2 f1p3 


"""
inv_freq_expanded = inv_freq[None, :, None]  # (1, 4, 1)

# Then expand to batch size:
inv_freq_expanded = inv_freq_expanded.expand(2, -1, 1)  # (2, 4, 1)

position_ids_expanded = position_ids[:, None, :]  # (2, 1, 3)