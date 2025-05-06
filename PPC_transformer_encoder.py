# this file defines transformer encoder layer with PPC and RIG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import math
import numpy as np
import copy
from PPC_linear import PPCLinear, update_all_gate_act_prob, get_all_rank_inference, get_all_orthogonality_loss, get_all_gate_act_prob, truncate_all_PPC, request_all_PPC

class PPCTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, bias = False, activation="relu", r_max_scale = 0.5):
        super(PPCTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.r_max = max(1, int(min(d_model, dim_feedforward) * r_max_scale))

        # Define the layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = PPCLinear(d_model, dim_feedforward, self.r_max, bias=bias)
        self.linear2 = PPCLinear(dim_feedforward, d_model, self.r_max, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            elif activation == "tanh":
                self.activation = F.tanh
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        
        x = src
        # Self-attention block
        x = self.norm1(
            x + self._sa_block(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        )
        # Feedforward block
        x = self.norm2(
            x + self._ff_block(x)
        )
        return x
        
    
    def _sa_block(self, x, attn_mask = None, key_padding_mask = None, is_causal = False):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal = is_causal)[0]
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.activation(self.linear1(x)))
        return self.dropout2(x)
    
class PPCTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(PPCTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal = False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            src = self.norm(src)
        return src
    
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Test the implementation
if __name__ == "__main__":
    # Example usage
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1
    batch_size = 32
    seq_length = 10

    # Define a PPC Transformer Encoder
    encoder_layer = PPCTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
    encoder= PPCTransformerEncoder(encoder_layer, num_layers = 2, norm = None)
    # Create a random input tensor
    src = torch.rand(seq_length, batch_size, d_model)
    src_mask = torch.zeros(seq_length, seq_length)
    src_key_padding_mask = torch.zeros(batch_size, seq_length)
    # Forward pass
    output = encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    print("Output shape:", output.shape)  # Should be (seq_length, batch_size, d_model)
    # Check rank inference
    ranks = get_all_rank_inference(encoder)
    print("Ranks:", ranks)
    # Check orthogonality loss
    orthogonality_loss = get_all_orthogonality_loss(encoder)
    print("Orthogonality loss:", orthogonality_loss)
    # Check gate activation probability
    gate_act_prob = get_all_gate_act_prob(encoder)
    print("Gate activation probability:", gate_act_prob)
    # Update gate activation probability
    update_all_gate_act_prob(encoder, 0.5)
    # Check updated gate activation probability
    gate_act_prob = get_all_gate_act_prob(encoder)
    print("Updated gate activation probability:", gate_act_prob)
    # Check truncation
    truncate_all_PPC(encoder)
    output = encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    print("Output shape after truncation:", output.shape)  # Should be (seq_length, batch_size, d_model)
    # Check rank inference after truncation
    ranks = get_all_rank_inference(encoder)
    print("Ranks after truncation:", ranks)

    