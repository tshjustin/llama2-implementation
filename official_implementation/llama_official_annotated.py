# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMS Norm is applied:

        1. Before the embeddings are passed to the GQA 

        2. After the embeddings are processed by the GQA and concatenatede with the skip link 

        Layer normalization 
        ---------------------
        Find statistic of each item / toekn in the sequence --> Works across the feature dimension 

        We use this in Llama since 

        a. Tokens should not be affected by other tokens 
        b. Indepdent of sequence length (Look at batch to see why it affects)


        Batch Normalization 
        ---------------------
        Normalizes across the batch dimension 

        Batch 1: [1.0, 2.0, 0.5, ...]  # Sample 1 features
        Batch 2: [0.8, 1.5, 0.3, ...]  # Sample 2 features  
        Batch 3: [1.2, 1.8, 0.7, ...]  # Sample 3 features
        Batch 4: [0.9, 2.1, 0.4, ...]  # Sample 4 features
                ↓    ↓    ↓
            Feature1 Feature2 Feature3

        Now normalize down the column 


        RMS Norm Formula 
        ----------------
        RMSNorm(x) = (x / RMS(x)) * γ,  RMS(x) = √(mean(x²) + ε), γ = learnable scaling parameter (self.weight)

        From findings, recentering dont contribute much, hence use scaling instead --> find some staistic that is independent of the mean of the token, hence we just the squares of x

        Each token is normalized from their own embeddings only 


        Data Flow - Input 
        ----------------
        batch_size = 2
        seq_len = 3  
        hidden_dim = 4

        hidden_states = torch.tensor(
            [
                [[1.0, 2.0, -1.0, 0.5], # Token 1, Batch 1
                [0.8, -0.3, 1.2, 0.1],  # Token 2, Batch 1  
                [0.2, 1.5, -0.8, 0.9]], # Token 3, Batch 1
                
                [[0.5, -1.0, 2.0, 0.3], # Token 1, Batch 2
                [1.1, 0.4, -0.5, 1.8],  # Token 2, Batch 2
                [-0.2, 0.7, 0.6, -1.1]] # Token 3, Batch 2
            ]
        )

        Data Flow - Calculate statistic  
        -------------------------------
        hidden_states.pow(2): Square every element
        .mean(-1, keepdim=True): Take mean along the last dimension (features)

        Input token: [1.0, 2.0, -1.0, 0.5] --> [1.0, 4.0, 1.0, 0.25]  -> .mean(-1) = (1.0+4.0+1.0+0.25)/4 = [1.5625] ==> RMS statistic for that token 

        rsqrt = 1/1.25 = 0.8

        
        Data Flow - Normalization   
        -------------------------------
        Normalized: [1.0*0.8, 2.0*0.8, -1.0*0.8, 0.5*0.8] = [0.8, 1.6, -0.8, 0.4]

        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype # keep initial data type 
        hidden_states = hidden_states.to(torch.float32) # float32 for rms calc to prevent precision errors 
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # compute variance 
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    """
    Rotary embedding is applied to Query and Key only

    Why?
    ----
    1. Attention scores is based on QK^T, rather than V, where we only use it for the output ==> hence use it for Q, K only 

    2. Position should only affect during attnetion calculation, rather than content retrieval --> absolute is actually abit wrong 

    3. From the slide, we can perfrom element wise operations of sin and cos, along with the pairs of inputs in the embedding vector 
    
    What does RoPE do?
    ------------------
    1. Compared to absolute, uses relative positions of tokens inside the sequence 

    2. Uses embedding pairs and perform rotation of each, based on some scaling of the angle  
    
    Code implementation 
    -------------------
    1. Generate rotation frequencies: inv_freq = [1/10000^(0/d), 1/10000^(2/d), 1/10000^(4/d), ...]

    2. Compute Position-Dependent Angles: freqs = position_ids @ inv_freq 

    # Position 0: [0×freq1, 0×freq2, ...] = [0, 0, ...]
    # Position 1: [1×freq1, 1×freq2, ...] = [freq1, freq2, ...]
    # Position 2: [2×freq1, 2×freq2, ...] = [2×freq1, 2×freq2, ...]
        
    3. Generate Cos/Sin for Rotation
    emb = torch.cat((freqs, freqs), dim=-1)  
    cos = emb.cos()  # Cosine components
    sin = emb.sin()  # Sine components

    4. Apply Rotation (in attention)


    How to implement RoPE 
    ---------------------
    1. For a word, the embedding dimesnion is d 
    
    2. Since we apply 2D rotations, we need to pair up each feature of the embedding ==> d // 2 pairs of features ==? d //2 number of frequencies 


    features = [f0, f1, f2, f3, f4, f5, f6, f7]

    pair_0 = [f0, f1] ->  Gets rotation frequency 1.0
    pair_1 = [f2, f3] ->  Gets rotation frequency 0.1 ...

    3. Think of each pair as a point in 2D space, which then gets rotated by an angle equal to that stated frequency 

    4. Each slower rotation could also capture different semantic meanings 


    5. Now extend the idea to a sentence "Cat sat on" 

    pair_0_cat = [f_cat_0, f_cat_1] = 0.0 
    pair_1_cat = [f_cat_2, f_cat_3] = 0.0 
    ...

    pair_0_sat = [f_sat_0, f_sat_1] = 1.0 
    pair_1_sat = [f_sat_2, f_sat_3] = 0.1 
    ...

    pair_0_on = [f_on_0, f_on_1] = 2.0  
    pair_1_on = [f_on_2, f_on_3] = 0.2
    ...

    We can combine all these information into a matrix 

    freq_pos_matrix = [
        [0.0, 1.0, 2.0],    # Angles for pair_0 at positions [0,1,2]
        [0.0, 0.1, 0.2],    # Angles for pair_1 at positions [0,1,2]  
        [0.0, 0.01, 0.02],  # Angles for pair_2 at positions [0,1,2]
        [0.0, 0.001, 0.002] # Angles for pair_3 at positions [0,1,2]
    ]

    Each row = rotation angles for one feature pair across all positions

    Each column = all rotation angles for one position
    
    6. Now that we got the rotation angles, we need to apply the actual rotation with trigonometric functionas 
    
    angles_cat = [0.0, 0.0, 0.0, 0.0]
    cos_cat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] --> Note that we have 8 entries here since [cos(0) cos(0) ...]
    sin_cat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                                            [sin(0) sin(0) ...] represents a rotation of a point --> hence matrix is 2x more  


    7. Now apply the actual rotation --> but using matrix multiplcation [x']   [cos(θ)  -sin(θ)] [x] is very expensive 
                                                                        [y'] = [sin(θ)   cos(θ)] [y]


    8. So convert the problem to a element wise operation (rotate_half)

        a. x_new = x * cos_theta - y * sin_theta --> this is much faster 
           y_new = x * sin_theta + y * cos_theta 

        b. 
        original = [x, y]
        rotate_half = [-y, x]  # Key transformation

        cos_duplicated = [cos, cos]
        sin_duplicated = [sin, sin]

        result = [x, y] * [cos, cos] + [-y, x] * [sin, sin]
            = [x*cos, y*cos] + [-y*sin, x*sin]
            = [x*cos - y*sin, y*cos + x*sin]
            = [x*cos - y*sin, x*sin + y*cos]  # Exactly the same as a. 

    0. Apply the embeddings 
    """
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type] 

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device) # retruns the inv_freq 

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):

        """
        (4, )
        self.inv_freq = [1.0, 0.1, 0.01, 0.001]
        
        # Add dimensions [None, :, None] → (1, 4, 1)
        temp = [
            [  # Batch dimension (size 1)
                [1.0],    # Frequency 0
                [0.1],    # Frequency 1  
                [0.01],   # Frequency 2
                [0.001]   # Frequency 3
            ]
        ]

        # Perform an expansion ->  Expand to batch size for example → (2, 4, 1)
        inv_freq_expanded = [
            [  # Batch 0
                [1.0], [0.1], [0.01], [0.001]
            ],
            [  # Batch 1  
                [1.0], [0.1], [0.01], [0.001]
            ]
        ]

        # Do the same for positions 
        Original: (2, 3)
        position_ids = [
            [0, 1, 2],  # Batch 0
            [0, 1, 2]   # Batch 1
        ]

        After [:, None, :] → (2, 1, 3)
        position_ids_expanded = [
            [  # Batch 0
                [0, 1, 2]  # Single row for broadcasting
            ],
            [  # Batch 1
                [0, 1, 2]  # Single row for broadcasting  
            ]
        ]

        # Now find the frquency matrix 
        (2, 4, 1) @ (2, 1, 3) → (2, 4, 3)
        freqs_before_transpose = [
            [  # Batch 0
                [0×1.0,   1×1.0,   2×1.0  ],  # [0.0,   1.0,   2.0  ]
                [0×0.1,   1×0.1,   2×0.1  ],  # [0.0,   0.1,   0.2  ]
                [0×0.01,  1×0.01,  2×0.01 ],  # [0.0,   0.01,  0.02 ]
                [0×0.001, 1×0.001, 2×0.001]   # [0.0,   0.001, 0.002]
            ],
            [  # Batch 1 (same values)
                [0.0,   1.0,   2.0  ],
                [0.0,   0.1,   0.2  ],
                [0.0,   0.01,  0.02 ],
                [0.0,   0.001, 0.002]
            ]
        ]


        # Duplicate along last dimension: (2, 3, 4) → (2, 3, 8) (cat) --> along some dimesnion --> last dimensioon in this case 
        emb = [
            [  # Batch 0
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],           # Position 0
                [1.0, 0.1, 0.01, 0.001, 1.0, 0.1, 0.01, 0.001],     # Position 1
                [2.0, 0.2, 0.02, 0.002, 2.0, 0.2, 0.02, 0.002]      # Position 2
            ],
            # Batch 1 same...
        ]

        # Now get the sin and cos matrix, attention scaling usally = 1 
        cos = [
            [  # Batch 0
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],                                    # cos(0) = 1
                [cos(1.0), cos(0.1), cos(0.01), cos(0.001), cos(1.0), cos(0.1), cos(0.01), cos(0.001)],  # Position 1
                [cos(2.0), cos(0.2), cos(0.02), cos(0.002), cos(2.0), cos(0.2), cos(0.02), cos(0.002)]   # Position 2
            ]
            # Batch 1 same...
        ]

        sin = [
            [  # Batch 0  
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                                    # sin(0) = 0
                [sin(1.0), sin(0.1), sin(0.01), sin(0.001), sin(1.0), sin(0.1), sin(0.01), sin(0.001)],  # Position 1
                [sin(2.0), sin(0.2), sin(0.02), sin(0.002), sin(2.0), sin(0.2), sin(0.02), sin(0.002)]   # Position 2
            ]
            # Batch 1 same...
        ]

        """ 
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) # exapand is a way to perform a copy 
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # frequency matirx 
            emb = torch.cat((freqs, freqs), dim=-1) # duplicate, since we have feature pairs --> note that a pair goes throigh the same rotation [cos] --> [cos cos]
            cos = emb.cos() * self.attention_scaling                                                                                           # [sin]     [sin, sin]
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """

    First half: [x₁, x₂, x₃, x₄] 
    
    Second half: [y₁, y₂, y₃, y₄]

    Result: [-y₁, -y₂, -y₃, -y₄, x₁, x₂, x₃, x₄]
    

    Q_cat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            [x₁,  y₁,  x₂,  y₂,  x₃,  y₃,  x₄,  y₄]
            ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^
            pair 1    pair 2    pair 3    pair 4

    rotate_half(Q_cat) = [-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]
                         [-y₁,  x₁,  -y₂,  x₂,  -y₃,  x₃,  -y₄,  x₄]
    
    cos_cat = [cos(1.0), cos(1.0), cos(0.1), cos(0.1), cos(0.01), cos(0.01), cos(0.001), cos(0.001)]
    sin_cat = [sin(1.0), sin(1.0), sin(0.1), sin(0.1), sin(0.01), sin(0.01), sin(0.001), sin(0.001)]   

    Use of rearrangment makes things very fast       

    x = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 elements
    x.shape[-1] = 8  # Size of the last dimension   


    x.shape[-1] // 2 = 8 // 2 = 4  # Half the size

    x.shape[-1] // 2 : = 4:  # "Start from index 4, go to the end"

    We are lucky that it is a 1d matrix --> so ... just means the above 

    """
    x1 = x[..., : x.shape[-1] // 2] # this synatx just means, take all items before the x.shape[-1]//2 index 
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """

    Consider 3 tokes

    cos =   [ (shape: 3, 8)
                [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000],
                [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000],
                [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000]
            ]

    sin =   [[0.8415, 0.8415, 0.0998, 0.0998, 0.0100, 0.0100, 0.0010, 0.0010],
            [0.8415, 0.8415, 0.0998, 0.0998, 0.0100, 0.0100, 0.0010, 0.0010],
            [0.8415, 0.8415, 0.0998, 0.0998, 0.0100, 0.0100, 0.0010, 0.0010]] 

    ## unsqueeze 

    cos_unsqueezed = [  (3, 1, 8)
                        [
                            [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000]
                        ],
                        [
                            [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000]
                        ],
                        [
                            [0.5403, 0.5403, 0.9950, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000]
                        ]
                    ]

    
    
    ## Assume that the q head looks like --> (2, 4, 3, 8)
    q = [
            [
                [
                    [q₀₀₀₀, q₀₀₀₁, q₀₀₀₂, q₀₀₀₃, q₀₀₀₄, q₀₀₀₅, q₀₀₀₆, q₀₀₀₇],    # batch 0, head 0, pos 0
                    [q₀₀₁₀, q₀₀₁₁, q₀₀₁₂, q₀₀₁₃, q₀₀₁₄, q₀₀₁₅, q₀₀₁₆, q₀₀₁₇],    # batch 0, head 0, pos 1
                    [q₀₀₂₀, q₀₀₂₁, q₀₀₂₂, q₀₀₂₃, q₀₀₂₄, q₀₀₂₅, q₀₀₂₆, q₀₀₂₇]     # batch 0, head 0, pos 2
                ],   
      
                [
                    [q₀₁₀₀, q₀₁₀₁, q₀₁₀₂, q₀₁₀₃, q₀₁₀₄, q₀₁₀₅, q₀₁₀₆, q₀₁₀₇],    # batch 0, head 1, pos 0
                    [q₀₁₁₀, q₀₁₁₁, q₀₁₁₂, q₀₁₁₃, q₀₁₁₄, q₀₁₁₅, q₀₁₁₆, q₀₁₁₇],    # batch 0, head 1, pos 1
                    [q₀₁₂₀, q₀₁₂₁, q₀₁₂₂, q₀₁₂₃, q₀₁₂₄, q₀₁₂₅, q₀₁₂₆, q₀₁₂₇]     # batch 0, head 1, pos 2
                ]
            ],  
     ...]]
    
     Broadcasting from right to left 
     q:   (2, 4, 3, 8)
     cos: (   3, 1, 8)
    
    cos gets exapnded to match ==> cos broadcasted: (2, 4, 3, 8)
    
    """
    cos = cos.unsqueeze(unsqueeze_dim) # (2, 3, 8) → (2, 1, 3, 8) (for example) --> adds a dimenson at the specified dim 
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    """
    Traditionally
    --------------
    hidden_layer --> output_layer || hidden_size → intermediate_size → hidden_size
    
    
    SwiGLU approach (3 layers with gating)
    --------------------------------------
    gate = activation(gate_proj(x))   # hidden_size → intermediate_size  
    up = up_proj(x)                   # hidden_size → intermediate_size
    output = down_proj(gate * up)     # intermediate_size → hidden_size
    

    Flow of information -- Example 
    -------------------------------
    x.shape = (batch_size, seq_len, hidden_size) = (2, 3, 4096) # Example: 2 sequences, 3 tokens each, 4096 features per token

    1. Parallel Projections 

    # Gate projection (will be activated)
    gate_output = self.gate_proj(x)  # (2, 3, 4096) → (2, 3, 11008)

    # Up projection (raw values) 
    up_output = self.up_proj(x)      # (2, 3, 4096) → (2, 3, 11008)


    2. Activation - Apply SiLU/Swish activation to gate
    activated_gate = self.act_fn(gate_output)  # (2, 3, 11008)


    3. Element-wise Multiplication (Gating)
    gated_values = activated_gate * up_output  # (2, 3, 11008) ⊙ (2, 3, 11008)

    
    4. Down Projection (Output)
    output = self.down_proj(gated_values)  # (2, 3, 11008) → (2, 3, 4096)


    Gating mechabnism makes more nuanced learning --> more complex inputs gets more attention from the MLP 

    same words used in different sentences gets different interpretation 

    Input → Gate Projection → Activation (SiLU/Swish)  ──┐
      │                                                  │
      └──→ Up Projection ────────────────────────────────┘
                                                        │
                                                        ▼
                                                Element-wise Product (*)
                                                        │
                                                        ▼
                                                Down Projection
                                                        │
                                                        ▼
                                                    Output


    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """

    Repeats the KV - heads 

    If we have 32 Query heads , but 8 Key/Value heads, then multiply by a factor of 4 


    Example 
    --------------
    2 Query heads, 1 Key/Value head
    n_rep = 2 // 1 = 2
    Sequence length = 2 tokens
    Head dimension = 3

    hidden_states shape: (1, 1, 2, 3)
    hidden_states = [
                        [
                            [
                                [1, 2, 3],      # KV head 0, token 0
                                [4, 5, 6]       # KV head 0, token 1
                            ]
                        ]
                    ]   

    1. Add new dimension --> hidden_states = hidden_states[:, :, None, :, :] --> (1, 1, 2, 3) → (1, 1, 1, 2, 3)

    hidden_states = [[[[[1, 2, 3],     # Batch 0, KV head 0, copy 0, token 0
                     [4, 5, 6]]]]]  # Batch 0, KV head 0, copy 0, token 1

    2. .exapnd(...) ==> repeat along new dimesnion ==> in this case .expand(1, 1, 2, 2, 3) --> Shape: (1, 1, 1, 2, 3) → (1, 1, 2, 2, 3)

    hidden_states = [[[[[1, 2, 3],     # Batch 0, KV head 0, copy 0, token 0
                     [4, 5, 6]],    # Batch 0, KV head 0, copy 0, token 1
                    
                    [[1, 2, 3],     # Batch 0, KV head 0, copy 1, token 0  
                     [4, 5, 6]]]]]  # Batch 0, KV head 0, copy 1, token 1

    3. .reshape(1, 2, 2, 3) - Flatten to Final Form (1, 1, 2, 2, 3) → (1, 2, 2, 3)

    hidden_states = [[[[1, 2, 3],      # Batch 0, Head 0, token 0
                   [4, 5, 6]],      # Batch 0, Head 0, token 1
                  
                  [[1, 2, 3],       # Batch 0, Head 1, token 0
                   [4, 5, 6]]]]     # Batch 0, Head 1, token 1
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    
    1. If we only have 4 key / value head, this means that the orignal embedding size = dim, would get split to dim // 2 --> different view of the matrix

    2. Hence during the repeate, we would repeat twice in this case to match, 

    3. Attention weights --> QK^T / sqrt(dim)
    
    4. Add in casual mask

    Masked attention scores:
            "The"  "cat"  "sat"
    "The"   [ 0.8  -inf  -inf ]
    "cat"   [ 0.4   0.9  -inf ]
    "sat"   [ 0.2   0.5   0.8 ]
    

    5. Apply softmax

    6. Apply dropouts 

    7. Multiply with values vector 



    Example 
    ---------------------

    1. Input 

    hidden_states shape: (1, 2, 8)  # batch=1, seq_len=2, hidden_size=8
    Token "cat": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    Token "dog": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

    2. Linear Projections (Q Projection in this example)

    Token "cat": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
    Token "dog": [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]

    
    3.   Reshape from (1, 2, 8) to (1, 2, 2, 4) --> reformat to 2 heads, hence size=4 for each 
    query_states = query_projection_output.view(1, 2, 2, 4) 

    Head 0 (dimensions 0-3):
    Token "cat": [2.1, 2.2, 2.3, 2.4]
    Token "dog": [3.1, 3.2, 3.3, 3.4]

    Head 1 (dimensions 4-7):  
    Token "cat": [2.5, 2.6, 2.7, 2.8]
    Token "dog": [3.5, 3.6, 3.7, 3.8]

    4. Transose for attention 
    query_states = query_states.transpose(1, 2)  # (1, 2, 2, 4) → (1, 2, 2, 4)


    Note
    ------------------
    1. "Sharding" is just getting a different view 
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # calculate attention scores 

    if attention_mask is not None:
        """
        Applies an attetion mask that affects the top right triangle 

        # Input sequence: "hi there i am"
        input_ids = [15, 456, 23, 891]  # Token IDs
        attention_mask = [1, 1, 1, 1]   # All real tokens (no padding) --> this is the very start, what we add to the model , that is different from the current attetnion_mask


        # Attetnion mask would look something like the below  --> initiated with max_position_embeddings  , hence very long 

        # Each "page" looks like a lower triangular matrix filled with 0.0 and -inf
        attention_mask[0, 0, :, :] = [
            [  0.0,  -inf,  -inf,  -inf, ..., -inf],  # Row 0
            [  0.0,   0.0,  -inf,  -inf, ..., -inf],  # Row 1
            [  0.0,   0.0,   0.0,  -inf, ..., -inf],  # Row 2
            [  0.0,   0.0,   0.0,   0.0, ..., -inf],  # Row 3
            ...
            [  0.0,   0.0,   0.0,   0.0, ...,  0.0]   # Row 127
        ]


        # In this case --> causal_mask would look something like  --> slicing to the current sequence length 
        causal_mask = [
            [
                [
                    [  0.0,  -inf,  -inf,  -inf],
                    [  0.0,   0.0,  -inf,  -inf], 
                    [  0.0,   0.0,   0.0,  -inf],
                    [  0.0,   0.0,   0.0,   0.0]
                ]
            ]
        ] Shape: (1, 1, 4, 4)
        
        
        """
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] # THis would do a trancuation --> attention_mask.shape = (1, 1, 2048, 2048)  --> attention_mask[:, :, :, :128]  # Take only first 128 columns
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)


    """
    In this case , we calculate the attetnino per head, then later on we concatenate 

    attn_output = [ (2, 8, 4, 64)
    # Batch 0
    [
        [[h0_pos0_vals...], [h0_pos1_vals...], [h0_pos2_vals...], [h0_pos3_vals...]],  # Head 0
        [[h1_pos0_vals...], [h1_pos1_vals...], [h1_pos2_vals...], [h1_pos3_vals...]],  # Head 1  
        [[h2_pos0_vals...], [h2_pos1_vals...], [h2_pos2_vals...], [h2_pos3_vals...]],  # Head 2
        [[h3_pos0_vals...], [h3_pos1_vals...], [h3_pos2_vals...], [h3_pos3_vals...]],  # Head 3
        [[h4_pos0_vals...], [h4_pos1_vals...], [h4_pos2_vals...], [h4_pos3_vals...]],  # Head 4
        [[h5_pos0_vals...], [h5_pos1_vals...], [h5_pos2_vals...], [h5_pos3_vals...]],  # Head 5
        [[h6_pos0_vals...], [h6_pos1_vals...], [h6_pos2_vals...], [h6_pos3_vals...]],  # Head 6
        [[h7_pos0_vals...], [h7_pos1_vals...], [h7_pos2_vals...], [h7_pos3_vals...]]   # Head 7
    ],
    # Batch 1
    [...]

    ## After transpose  (2, 4, 8, 64) --> swop positions 
    attn_output = [
    # Batch 0
        [
            [[h0_pos0_vals...], [h1_pos0_vals...], [h2_pos0_vals...], ..., [h7_pos0_vals...]],  # Position 0, all heads
            [[h0_pos1_vals...], [h1_pos1_vals...], [h2_pos1_vals...], ..., [h7_pos1_vals...]],  # Position 1, all heads
            [[h0_pos2_vals...], [h1_pos2_vals...], [h2_pos2_vals...], ..., [h7_pos2_vals...]],  # Position 2, all heads  
            [[h0_pos3_vals...], [h1_pos3_vals...], [h2_pos3_vals...], ..., [h7_pos3_vals...]]   # Position 3, all heads
        ],
        # Batch 1
        [...]
    ]
    
    """
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """

    Dimensions of heads 
    -----------------------

    Query
    ======
    If we have 32 number of heads, then we would have 32 query heads 

    For a dimension size of 4096, each query head would thus have (4096 / 32) dimension size 

    Key Value 
    ==========
    Note that we preserve the same number of dimension as per the query heads => (4096 / 32)

    But instead of the 32 heads, each having (4096 / 32) dimensions 

    We have n_kv_head heads, each having (4096 / 32) dimensions ==> Much lesser number of parameters 

    Example
    ================

    Token 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    Token 1: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    1. Query Projection (8 → 8 dimensions)
    Token 0 Q: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    Token 1 Q: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    2. Key Projection (8 → 4 dimensions) --> Since we only have 2 KV heads here --> Lesser heads to represent the SAME NUMBER OF DIMENSIONS ==> Lead to overall reduction in parameter count 
    Token 0 K: [0.2, 0.4, 0.6, 0.8]
    Token 1 K: [0.4, 0.6, 0.8, 1.0]

    3. Value Projection (8 → 4 dimensions)
    Token 0 V: [0.3, 0.6, 0.9, 1.2] 
    Token 1 V: [0.6, 0.9, 1.2, 1.5]
    
    4. Split to query Heads (1, 2, 8) → (1, 4, 2, 2)
    Query Head 0: Token 0: [0.1, 0.2]    Token 1: [0.2, 0.3]
    Query Head 1: Token 0: [0.3, 0.4]    Token 1: [0.4, 0.5] 
    Query Head 2: Token 0: [0.5, 0.6]    Token 1: [0.6, 0.7]
    Query Head 3: Token 0: [0.7, 0.8]    Token 1: [0.8, 0.9]


    5. Split to KV heads 
    Key States: (1, 2, 4) → (1, 2, 2, 2)
    # Key 
    KV Head 0: Token 0: [0.2, 0.4]    Token 1: [0.4, 0.6]
    KV Head 1: Token 0: [0.6, 0.8]    Token 1: [0.8, 1.0]

    # Values 
    KV Head 0: Token 0: [0.3, 0.6]    Token 1: [0.6, 0.9]
    KV Head 1: Token 0: [0.9, 1.2]    Token 1: [1.2, 1.5]

    6. Repeat KV heads, for both value and key  
    Key Head 0: Token 0: [0.2, 0.4]    Token 1: [0.4, 0.6]  ← Copy of KV Head 0
    Key Head 1: Token 0: [0.2, 0.4]    Token 1: [0.4, 0.6]  ← Copy of KV Head 0  
    Key Head 2: Token 0: [0.6, 0.8]    Token 1: [0.8, 1.0]  ← Copy of KV Head 1
    Key Head 3: Token 0: [0.6, 0.8]    Token 1: [0.8, 1.0]  ← Copy of KV Head 1

    7. Attention QK^T --> Similar to MHA 
    Q_head0 = [[0.1, 0.2], [0.2, 0.3]]  # 2 tokens × 2 dims
    K_head0 = [[0.2, 0.4], [0.4, 0.6]]  # 2 tokens × 2 dims

    Q × K^T = [[0.1, 0.2], [0.2, 0.3]] × [[0.2, 0.4], [0.4, 0.6]]^T
            = [[0.1, 0.2], [0.2, 0.3]] × [[0.2, 0.4], [0.4, 0.6]]

    8. Apply casual mask
    Masked scores = [[0.071, -inf], 
                    [0.113, 0.184]]

    9. Soft max and multiply with V 

    10. Concate all the heads -> (1, 2, 8)
    Token 0 output: [0.3, 0.6, 0.3, 0.6, 0.9, 1.2, 0.9, 1.2]
    Token 1 output: [0.455, 0.755, 0.455, 0.755, 1.365, 1.755, 1.365, 1.755]

    11. Final output projection 
    Token 0: [0.15, 0.3, 0.15, 0.3, 0.45, 0.6, 0.45, 0.6]
    Token 1: [0.228, 0.378, 0.228, 0.378, 0.683, 0.878, 0.683, 0.878]

    ==========
    KV Cache 
    ==========

    past_key_values = None  # No cache yet
    hidden_states = "The" embedding: (1, 1, 8)

    Forward Pass 1: 
    --------------------
    key_states = self.k_proj(hidden_states)    # (1, 1, 4)
    value_states = self.v_proj(hidden_states)  # (1, 1, 4)

    # Reshape to heads  
    key_states = key_states.view(1, 1, 2, 2).transpose(1, 2)    # (1, 2, 1, 2)
    value_states = value_states.view(1, 1, 2, 2).transpose(1, 2) # (1, 2, 1, 2)

    Key states shape: (1, 2, 1, 2)
    KV Head 0: Token "The": [0.2, 0.4]
    KV Head 1: Token "The": [0.6, 0.8]

    Value states shape: (1, 2, 1, 2)  
    KV Head 0: Token "The": [0.3, 0.6]
    KV Head 1: Token "The": [0.9, 1.2]

    Apply RoPE
    cos, sin = position_embeddings  # Position 0 for "The"
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    After RoPE
    Key states:
    KV Head 0: Token "The": [0.21, 0.39]
    KV Head 1: Token "The": [0.61, 0.79]

    Since no cache at first --> after the first attention calculation --> would create the cache 
    past_key_values.update(key_states, value_states, layer_idx=0, cache_kwargs)

    past_key_values[layer_0]:
    keys:   (1, 2, 1, 2) - 2 KV heads, 1 token, 2 dims per head
    values: (1, 2, 1, 2)
    
    KV Head 0: Key=[0.21, 0.39], Value=[0.3, 0.6]  # Token "The"
    KV Head 1: Key=[0.61, 0.79], Value=[0.9, 1.2]  # Token "The"


    Forward Pass 2
    -----------------------
    past_key_values = <contains "The" data>  # Cache exists now
    hidden_states = "cat" embedding: (1, 1, 8)  # Only new token 

    # Linear projections for "cat" only --> Note that previously without KV cache we would have calculate the same for "Cat"
    key_states = self.k_proj(hidden_states)    # (1, 1, 4)
    value_states = self.v_proj(hidden_states)  # (1, 1, 4)

    # Reshape to heads
    key_states = key_states.view(1, 1, 2, 2).transpose(1, 2)    # (1, 2, 1, 2)
    value_states = value_states.view(1, 1, 2, 2).transpose(1, 2) # (1, 2, 1, 2)

    ... same operations ... 

    NOTE: Now would save the KV embeddings for CAT 
    if past_key_values is not None: 
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": torch.tensor([1])}
        key_states, value_states = past_key_values.update(
            key_states,      # New keys for "cat": (1, 2, 1, 2)
            value_states,    # New values for "cat": (1, 2, 1, 2)  
            self.layer_idx,  # layer_idx = 0
            cache_kwargs
        )

    past_key_values[layer_0]:
        keys:   (1, 2, 2, 2) - 2 KV heads, 2 tokens, 2 dims per head
        values: (1, 2, 2, 2)

    ...

    Layer 0 Key Cache:
    [
        batch_0: [
            head_0: [[k₁₁, k₁₂, k₁₃, k₁₄], [k₂₁, k₂₂, k₂₃, k₂₄], [k₃₁, k₃₂, k₃₃, k₃₄]], ==> stores the sharded matrics 
            head_1: [[k₁₁, k₁₂, k₁₃, k₁₄], [k₂₁, k₂₂, k₂₃, k₂₄], [k₃₁, k₃₂, k₃₃, k₃₄]],
            head_2: [[k₁₁, k₁₂, k₁₃, k₁₄], [k₂₁, k₂₂, k₂₃, k₂₄], [k₃₁, k₃₂, k₃₃, k₃₄]],
            head_3: [[k₁₁, k₁₂, k₁₃, k₁₄], [k₂₁, k₂₂, k₂₃, k₂₄], [k₃₁, k₃₂, k₃₃, k₃₄]]
        ]
    ]


    ==============================
    How the actual cache looks like
    ==============================
    Consider the cases only after RoPE 

    rotated_keys = [
        head_0: [0.42, -0.18, 0.75, 0.15, ..., 0.68],   # Slightly modified by RoPE
        head_1: [0.31, -0.48, 0.88, -0.08, ..., 0.18]
    ] 

    DynamicCache {
        key_cache: [
            # Layer 0 keys
            tensor([[[[ 0.42, -0.18,  0.75,  0.15, ...,  0.68],    # head 0, token 1 ------------> In this example, there are 3 layrs. These layers are applied sequenially, one token at a time 
                    [ 0.31, -0.48,  0.88, -0.08, ...,  0.18]]]]), # head 1, token 1  ------------> the proper fill order is Layer 0 --> layer 0 + layer 1 --> layer 0 + layer 1 + layer 2 
            
            # Layer 1 keys  
            tensor([[[[ 0.73, -0.21,  0.45,  0.67, ...,  0.12],    # head 0, token 1
                    [ 0.89, -0.34,  0.56, -0.23, ...,  0.91]]]]), # head 1, token 1
                    
            # Layer 2 keys
            tensor([[[[ 0.34, -0.67,  0.78,  0.23, ...,  0.45],    # head 0, token 1  
                    [ 0.12, -0.89,  0.34,  0.67, ...,  0.78]]]])  # head 1, token 1
        ],
        
        value_cache: [
            # Layer 0 values (similar structure to keys but different values)
            tensor([[[[ 0.78, -0.23,  0.45,  0.67, ...,  0.12],
                    [ 0.34, -0.67,  0.78,  0.23, ...,  0.45]]]]),
            # Layer 1 values
            tensor([[[[ 0.56, -0.78,  0.34,  0.12, ...,  0.89],
                    [ 0.67, -0.23,  0.45,  0.78, ...,  0.34]]]]),
            # Layer 2 values  
            tensor([[[[ 0.89, -0.34,  0.56,  0.67, ...,  0.23],
                    [ 0.45, -0.78,  0.12,  0.89, ...,  0.56]]]])
        ],
        
        _seen_tokens: 1
    }

    ## Now for the next word

    new_key_states = [
        head_0: [0.67, -0.34, 0.12, 0.89, ..., 0.23],   # 32 dims for token 2
        head_1: [0.45, -0.78, 0.56, -0.23, ..., 0.67]   # 32 dims for token 2  
    ]
    

    DynamicCache {
        key_cache: [
            # Layer 0 keys - NOW WITH 2 TOKENS
            tensor([[[[ 0.42, -0.18,  0.75,  0.15, ...,  0.68],    # head 0, token 1
                    [ 0.67, -0.34,  0.12,  0.89, ...,  0.23]],   # head 0, token 2
                    [[ 0.31, -0.48,  0.88, -0.08, ...,  0.18],    # head 1, token 1  
                    [ 0.45, -0.78,  0.56, -0.23, ...,  0.67]]]]), # head 1, token 2
            
            # Layer 1 keys - ALSO WITH 2 TOKENS
            tensor([[[[ 0.73, -0.21,  0.45,  0.67, ...,  0.12],    # head 0, token 1
                    [ 0.23, -0.56,  0.78,  0.34, ...,  0.89]],   # head 0, token 2
                    [[ 0.89, -0.34,  0.56, -0.23, ...,  0.91],    # head 1, token 1
                    [ 0.12, -0.67,  0.34,  0.78, ...,  0.45]]]]), # head 1, token 2
                    
            # Layer 2 keys - ALSO WITH 2 TOKENS  
            tensor([[[[ 0.34, -0.67,  0.78,  0.23, ...,  0.45],    # head 0, token 1
                    [ 0.89, -0.12,  0.56,  0.67, ...,  0.34]],   # head 0, token 2
                    [[ 0.12, -0.89,  0.34,  0.67, ...,  0.78],    # head 1, token 1
                    [ 0.78, -0.23,  0.45,  0.89, ...,  0.12]]]])  # head 1, token 2
        ],
        
        value_cache: [
            # Similar structure - each layer now has 2 tokens worth of values
            # Shape for each: [1, 2, 2, 32]
            ...
        ],
        
        _seen_tokens: 2
    }

    Now when we calculate attention --> 
    1. Query --> Takes from all heads , but only the most recent token 

    2. Key --> Takes from all heads, for all the tokens 


    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # projections ->  2048 → 32 × 64 = 2048
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )

        # # 2048 → 8 × 64 = 512 , since we are performing GQA 
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        # concatenate back to input 
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # linear projections 
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # positional embeddings 
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handles KV cache 
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # eager / flash_attention_z / sdpa 
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # compute attention 
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # flatten all the head together 
        """
        attn_output[0, 0, :, :] = 
        [
            [head0_64_values...],  # Head 0 output for position 0
            [head1_64_values...],  # Head 1 output for position 0  
            [head2_64_values...],  # Head 2 output for position 0
            ...
            [head7_64_values...]   # Head 7 output for position 0
        ]
            
        
        # Before: Separate heads
        Position 0: [[H0], [H1], [H2], [H3], [H4], [H5], [H6], [H7]]
        Position 1: [[H0], [H1], [H2], [H3], [H4], [H5], [H6], [H7]]
        ...

        # After: Concatenated heads  
        Position 0: [H0|H1|H2|H3|H4|H5|H6|H7]  # 512-dim vector
        Position 1: [H0|H1|H2|H3|H4|H5|H6|H7]  # 512-dim vector
        ...
        


        ### Pytorch operations 

        input_shape = (2, 4) ==>. unpack == 2, 4

        .reshape(2, 4 , -1).contiguous()

        # Before reshape: (batch, seq, heads, head_dim)
        attn_output.shape = (2, 4, 8, 64)

        # reshape(2, 4, -1) ==> keep the first 2 dimentsion ==> fix the last dimentison 
        # The -1 means "calculate this dimension automatically"
        # -1 = (8 * 64) = 512
        attn_output.shape = (2, 4, 512)

        .contiguous() ==> ensures that memory are sequnetially and not scattered 
        """
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # final projection layer 
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor, #  main tensor flowing through the layer

        attention_mask: Optional[torch.Tensor] = None, 

        position_ids: Optional[torch.LongTensor] = None, # [0, 1, 2, 3, ...], there is batch size here 

        past_key_values: Optional[Cache] = None, # kv cache 

        use_cache: Optional[bool] = False,
        
        cache_position: Optional[torch.LongTensor] = None,

        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn( # prviosuly layer embeddings   --> though in the image, there is an addition of RMS norm after initla embeddings --> but no additional embeddings after the first decoder output --> jsut decoder_layer_1 as inputs to the next layer 
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states 

        # Fully Connected --> just go through the blocks 
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    """
    Base model wihtout any heads for any task   
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """
    Language modelling head 
    
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            """
            This is for training --> for training of llama2
            
            """
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


"""
For other tasks --> ... in a class means to inherit everything as is 

For classification head --> can be found in GenericForSequenceClassification 

"""
class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]