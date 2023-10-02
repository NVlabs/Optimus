# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection
"""

import math
import textwrap
from collections import OrderedDict

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import torch.distributions as D
import torch.nn as nn
from robomimic.models.base_nets import Module
from robomimic.models.distributions import TanhWrappedDistribution
from robomimic.models.vae_nets import CategoricalPrior, GaussianPrior
from torch.nn import functional as F
from torch.optim import Optimizer

from tamp_imitation.models.obs_nets import (
    MIMO_MLP,
    ObservationDecoder,
    ObservationGroupEncoder,
)


def freeze_net(net):
    net.eval()
    for param in net.parameters():
        param.requires_grad = False


class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    Implementation: https://github.com/pfnet-research/deep-table/blob/237c8be8a405349ce6ab78075234c60d9bfe60b7/deep_table/nn/layers/activation.py
    """

    def geglu(self, x):
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
        Implementation taken from https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group["lr"]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim):
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


class CausalSelfAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attention_dropout=0.1,
        output_dropout=0.1,
    ):
        """
        Multi-head masked self-attention layer + projection (MLP layer).

        For normal self-attention (@num_heads = 1), every single input in the sequence is
        mapped to a key, query, and value embedding of size @embed_dim. For each input,
        its query vector is compared (using dot-product) with all other key vectors in the
        sequence, and softmax normalized to compute an attention over all members of the
        sequence. This is used to take a linear combination of corresponding value embeddings.

        The @num_heads argument is for multi-head attention, where the self-attention operation above
        is performed in parallel over equal size partitions of the @embed_dim, allowing for different
        portions of the embedding dimension to model different kinds of attention. The attention
        output for each head is concatenated together.

        Finally, we use a causal mask here to ensure that each output only depends on inputs that come
        before it.

        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attention_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs
        """
        super(CausalSelfAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["qkv"] = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        # dropout layers
        self.nets["attention_dropout"] = nn.Dropout(self.attention_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)

    def forward(self, x, unused):
        """
        Forward pass through Self-Attention block.
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """

        # enforce shape consistency
        assert len(x.shape) == 3
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim
        NH = self.num_heads  # number of attention heads
        DH = D // NH  # embed dimension for each attention head

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        qkv = self.nets["qkv"](x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        q = q.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        v = v.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]

        # causal self-attention mechanism

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, NH, T, DH] x [B, NH, DH, T] -> [B, NH, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.
        att = att.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        att = F.softmax(
            att, dim=-1
        )  # shape [B, NH, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        att = self.nets["attention_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, NH, T, T] x [B, NH, T, DH] -> [B, NH, T, DH]
        y = att @ v
        # reshape [B, NH, T, DH] -> [B, T, NH, DH] -> [B, T, NH * DH] = [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        return y

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class CausalCrossAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attention_dropout=0.1,
        output_dropout=0.1,
        key_value_from_condition=False,
    ):
        """
        Multi-head masked self-attention layer + projection (MLP layer).

        For normal self-attention (@num_heads = 1), every single input in the sequence is
        mapped to a key, query, and value embedding of size @embed_dim. For each input,
        its query vector is compared (using dot-product) with all other key vectors in the
        sequence, and softmax normalized to compute an attention over all members of the
        sequence. This is used to take a linear combination of corresponding value embeddings.

        The @num_heads argument is for multi-head attention, where the self-attention operation above
        is performed in parallel over equal size partitions of the @embed_dim, allowing for different
        portions of the embedding dimension to model different kinds of attention. The attention
        output for each head is concatenated together.

        Finally, we use a causal mask here to ensure that each output only depends on inputs that come
        before it.

        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attention_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs

            key_value_from_condition (bool): if True, compute keys and values from the condition
        """
        super(CausalCrossAttention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), "num_heads: {} does not divide embed_dim: {} exactly".format(num_heads, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.key_value_from_condition = key_value_from_condition
        self.nets = nn.ModuleDict()

        # projection layers for key, query, value, across all attention heads
        self.nets["kv"] = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        self.nets["q"] = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # dropout layers
        self.nets["attention_dropout"] = nn.Dropout(self.attention_dropout)
        self.nets["output_dropout"] = nn.Dropout(self.output_dropout)

        # output layer
        self.nets["output"] = nn.Linear(self.embed_dim, self.embed_dim)

        # causal mask (ensures attention is only over previous inputs) - just a lower triangular matrix of 1s
        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("mask", mask)

    def forward(self, x, condition):
        """
        Forward pass through Self-Attention block.
        Input should be shape (B, T, D) where B is batch size, T is seq length (@self.context_length), and
        D is input dimension (@self.embed_dim).
        """
        # enforce shape consistency
        assert (
            x.shape == condition.shape
        ), "Input shape: {} does not match condition shape: {}".format(x.shape, condition.shape)
        assert len(x.shape) == 3
        B, T, D = x.shape
        assert (
            T <= self.context_length
        ), "self-attention module can only handle sequences up to {} in length but got length {}".format(
            self.context_length, T
        )
        assert D == self.embed_dim
        NH = self.num_heads  # number of attention heads
        DH = D // NH  # embed dimension for each attention head

        # compute key, query, and value vectors for each member of sequence, and split across attention heads
        if self.key_value_from_condition:
            kv = self.nets["kv"](condition)
            q = self.nets["q"](x)
        else:
            kv = self.nets["kv"](x)
            q = self.nets["q"](condition)

        k, v = torch.chunk(kv, 2, dim=-1)
        k = k.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        v = v.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]
        q = q.view(B, T, NH, DH).transpose(1, 2)  # [B, NH, T, DH]

        # causal self-attention mechanism

        # batched matrix multiplication between queries and keys to get all pair-wise dot-products.
        # We broadcast across batch and attention heads and get pair-wise dot-products between all pairs of timesteps
        # [B, NH, T, DH] x [B, NH, DH, T] -> [B, NH, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # use mask to replace entries in dot products with negative inf to ensure they don't contribute to softmax,
        # then take softmax over last dimension to end up with attention score for each member of sequence.
        # Note the use of [:T, :T] -  this makes it so we can handle sequences less than @self.context_length in length.
        att = att.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        att = F.softmax(
            att, dim=-1
        )  # shape [B, NH, T, T], last dimension has score over all T for each sequence member

        # dropout on attention
        att = self.nets["attention_dropout"](att)

        # take weighted sum of value vectors over whole sequence according to attention, with batched matrix multiplication
        # [B, NH, T, T] x [B, NH, T, DH] -> [B, NH, T, DH]
        y = att @ v
        # reshape [B, NH, T, DH] -> [B, T, NH, DH] -> [B, T, NH * DH] = [B, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # pass through output layer + dropout
        y = self.nets["output"](y)
        y = self.nets["output_dropout"](y)
        return y

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class Transformer_Block(Module):
    """
    A single Transformer Block, that can be chained together repeatedly.
    It consists of a @CausalSelfAttention module and a small MLP, along with
    layer normalization and residual connections on each input.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        attention_dropout=0.1,
        output_dropout=0.1,
        activation=nn.GELU(),
        drop_path=0.0,
        use_cross_attention_conditioning=False,
        use_alternating_cross_attention_conditioning=False,
        key_value_from_condition=False,
        layer_index=0,
    ):
        """
        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            context_length (int): expected length of input sequences

            attention_dropout (float): dropout probability for attention outputs

            output_dropout (float): dropout probability for final outputs

            activation (str): string denoting the activation function to use in each transformer block

            drop_path (float): dropout probability to use in drop path

            use_cross_attention_conditioning (bool): whether to use cross attention conditioning

            use_alternating_cross_attention_conditioning (bool): whether to use alternating cross attention conditioning

            key_value_from_condition (bool): if True, compute keys and values from the condition

            layer_index (int): index of the layer
        """
        super(Transformer_Block, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.use_cross_attention_conditioning = use_cross_attention_conditioning
        self.nets = nn.ModuleDict()

        # self-attention block
        if use_cross_attention_conditioning:
            self.nets["attention"] = CausalCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=context_length,
                attention_dropout=attention_dropout,
                output_dropout=output_dropout,
                key_value_from_condition=key_value_from_condition,
            )
        elif use_alternating_cross_attention_conditioning:
            if layer_index % 2 == 0:
                self.nets["attention"] = CausalCrossAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    attention_dropout=attention_dropout,
                    output_dropout=output_dropout,
                    key_value_from_condition=key_value_from_condition,
                )
            else:
                self.nets["attention"] = CausalSelfAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    attention_dropout=attention_dropout,
                    output_dropout=output_dropout,
                )
        else:
            self.nets["attention"] = CausalSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=context_length,
                attention_dropout=attention_dropout,
                output_dropout=output_dropout,
            )

        # self.nets["attention"] = FlashMHA(
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        #     context_length=context_length,
        #     attention_dropout=attention_dropout,
        #     output_dropout=output_dropout,
        #     causal=True,
        #     device='cuda'
        # )

        if type(activation) == GEGLU:
            mult = 2
        else:
            mult = 1

        # small 2-layer MLP
        self.nets["mlp"] = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim * mult),
            activation,
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(output_dropout),
        )

        # layer normalization for inputs to self-attention module and MLP
        self.nets["ln1"] = nn.LayerNorm(embed_dim)
        self.nets["ln2"] = nn.LayerNorm(embed_dim)
        self.nets["drop_path1"] = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.nets["drop_path2"] = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, inputs):
        """
        Forward pass - chain self-attention + MLP blocks, with residual connections and layer norms.
        """
        # x = x + self.nets["attention"](self.nets["ln1"](x), None, None)[0]
        x, condition = inputs["x"], inputs["condition"]
        x = x + self.nets["drop_path1"](self.nets["attention"](self.nets["ln1"](x), condition))
        x = x + self.nets["drop_path2"](self.nets["mlp"](self.nets["ln2"](x)))
        return {"x": x, "condition": condition}

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module doesn't modify the size of the input, it goes from (B, T, D) -> (B, T, D)
        return list(input_shape)


class GPT_Backbone(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        embed_dim,
        context_length,
        block_attention_dropout=0.1,
        block_output_dropout=0.1,
        block_drop_path=0.0,
        num_layers=6,
        num_heads=8,
        use_custom_transformer_block=True,
        activation="gelu",
        use_cross_attention_conditioning=False,
        use_alternating_cross_attention_conditioning=False,
        key_value_from_condition=False,
    ):
        """
        Args:
            embed_dim (int): dimension of embeddings to use for keys, queries, and values
                used in self-attention

            context_length (int): expected length of input sequences

            block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            block_output_dropout (float): dropout probability for final outputs for each transformer block

            num_layers (int): number of transformer blocks to stack

            num_heads (int): number of attention heads - must divide @embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            use_custom_transformer_block (bool): if True, use custom transformer block

            activation (str): string denoting the activation function to use in each transformer block

            use_cross_attention_conditioning (bool): whether to use cross attention conditioning

            use_alternating_cross_attention_conditioning (bool): whether to use alternating cross attention conditioning

        """
        super(GPT_Backbone, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.block_attention_dropout = block_attention_dropout
        self.block_output_dropout = block_output_dropout
        self.use_custom_transformer_block = use_custom_transformer_block
        self.block_drop_path = block_drop_path
        self.use_cross_attention_conditioning = use_cross_attention_conditioning
        self.use_alternating_cross_attention_conditioning = (
            use_alternating_cross_attention_conditioning
        )
        self.key_value_from_condition = key_value_from_condition

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "geglu":
            self.activation = GEGLU()

        # create networks
        self._create_networks()

        # initialize weights
        self.apply(self._init_weights)

        print(
            "Created {} model with number of parameters: {}".format(
                self.__class__.__name__, sum(p.numel() for p in self.parameters())
            )
        )

    def _create_networks(self):
        """
        Helper function to create networks.
        """
        self.nets = nn.ModuleDict()

        # transformer - cascaded transformer blocks
        if self.use_custom_transformer_block:
            self.nets["transformer"] = nn.Sequential(
                *[
                    Transformer_Block(
                        embed_dim=self.embed_dim,
                        num_heads=self.num_heads,
                        context_length=self.context_length,
                        attention_dropout=self.block_attention_dropout,
                        output_dropout=self.block_output_dropout,
                        activation=self.activation,
                        drop_path=self.block_drop_path,
                        use_cross_attention_conditioning=self.use_cross_attention_conditioning,
                        use_alternating_cross_attention_conditioning=self.use_alternating_cross_attention_conditioning,
                        key_value_from_condition=self.key_value_from_condition,
                        layer_index=i,
                    )
                    for i in range(self.num_layers)
                ]
            )
        else:
            self.nets["transformer"] = nn.Sequential(
                *[
                    torch.nn.TransformerEncoderLayer(
                        d_model=self.embed_dim,
                        nhead=self.num_heads,
                        dim_feedforward=4 * self.embed_dim,
                        norm_first=True,
                        batch_first=True,
                        dropout=self.block_attention_dropout,
                        activation=self.activation,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        # decoder head
        self.nets["output_ln"] = nn.LayerNorm(self.embed_dim)

    def _init_weights(self, module):
        """
        Weight initializer.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module takes inputs (B, T, @self.input_dim) and produces outputs (B, T, @self.output_dim)
        return input_shape[:-1] + [self.output_dim]

    def forward(self, inputs, condition=None):
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), inputs.shape
        x = self.nets["transformer"]({"x": inputs, "condition": condition})["x"]
        transformer_output = self.nets["output_ln"](x)
        return transformer_output


class MEGA_Backbone(nn.Module):
    """the full MEGA encoder model, with a context size of block_size"""

    def __init__(
        self,
        context_length,
        embed_dim,
        num_layers,
        mega_kwargs=None,
    ):
        """
        Args:


        """
        super(MEGA_Backbone, self).__init__()

        if mega_kwargs is None:
            mega_kwargs = {}
        mega_kwargs["embedding_dim"] = embed_dim
        mega_kwargs["num_encoder_layers"] = num_layers
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.nets = nn.ModuleDict()
        self.nets["transformer"] = MegaEncoder(**mega_kwargs)

        print(
            "Created {} model with number of parameters: {}".format(
                self.__class__.__name__, sum(p.numel() for p in self.parameters())
            )
        )

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # this module takes inputs (B, T, @self.input_dim) and produces outputs (B, T, @self.output_dim)
        return input_shape[:-1] + [self.output_dim]

    def forward(self, inputs, condition=None):
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), inputs.shape
        x = self.nets["transformer"](inputs, last_state_only=True)[0].transpose(0, 1)
        return x


class MIMO_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as
    a dictionary of observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_embedding_dropout=0.1,
        transformer_block_attention_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_block_drop_path=0.0,
        transformer_relative_timestep=False,
        transformer_max_timestep=1250,
        transformer_euclidean_distance_timestep=False,
        transformer_sinusoidal_embedding=False,
        transformer_use_custom_transformer_block=True,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        transformer_task_id_embed_dim=0,
        transformer_num_task_ids=1,
        transformer_language_enabled=False,
        transformer_language_embedding="raw",
        transformer_finetune_language_embedding=False,
        transformer_language_as_task_id=True,
        transformer_decoder=False,
        transformer_primitive_type="none",
        transformer_use_cross_attention_conditioning=False,
        transformer_use_alternating_cross_attention_conditioning=False,
        transformer_key_value_from_condition=False,
        transformer_add_primitive_id=False,
        transformer_tokenize_primitive_id=False,
        transformer_channel_condition=False,
        transformer_tokenize_obs_components=False,
        transformer_num_patches_per_image_dim=1,
        transformer_nets_to_freeze=(),
        transformer_use_ndp_decoder=False,
        transformer_ndp_decoder_kwargs=None,
        transformer_type="gpt",
        mega_kwargs=None,
        use_cvae=True,
        predict_signature=False,
        layer_dims=(1024, 1024),
        latent_dim=16,
        prior_use_gmm=True,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=True,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        replan_every_step=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block

            transformer_relative_timestep (bool): if True, timesteps range from 0 to context length - 1, if False use absolute position in trajectory.

            transformer_max_timestep (int): when using absolute timesteps or euclidean distance timesteps,
                define the maximal timestep value

            transformer_euclidean_distance_timestep (int): if True, use cumulative end-effector distance traveled as timesteps

            transformer_sinusoidal_embedding (bool): if True, use sinusoidal positional embeddings that are not learned

            transformer_use_custom_transformer_block (bool): if True, use custom transformer block

            transformer_activation (str): string denoting the activation function to use in each transformer block

            transformer_nn_parameter_for_timesteps (bool): if True, use nn.Parameter for embedding timesteps

            transformer_task_id_embed_dim (int): use nn.Embedding to embed task ids

            transformer_num_task_ids (int): number of tasks we are training with

            transformer_language_enabled (bool): if True, condition on language embeddings

            transformer_language_embedding (str): string denoting the language embedding to use

            transformer_finetune_language_embedding (bool): if True, finetune the language embedding

            transformer_primitive_type (str): string denoting the primitive id to use

            transformer_use_cross_attention_conditioning (bool): if True, use cross attention conditioning

            transformer_use_alternating_cross_attention_conditioning (bool): if True, use alternating cross attention conditioning

            transformer_tokenize_obs_components (bool): if True, tokenize observation components

            transformer_nets_to_freeze (tuple): tuple of strings denoting which nets to freeze

            transformer_use_ndp_decoder (bool): if True, use NDP decoder

            use_cvae (bool): if True, use condition on initial obs for the prior and encoder

            predict_signature (bool): if True, instead of VIB, use latents from the encoder to predict the signature
            and condition transformer

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

            latent_dim (int): dimension of latent space for the VAE

            prior_use_gmm (bool): if True, learn a Gaussian Mixture Model (GMM)
                prior instead of a unimodal Gaussian prior.

            prior_gmm_num_modes (int): number of GMM modes to learn. Only
                used if @prior_use_gmm is True.

            prior_gmm_learn_weights (bool): if True, learn the weights of the GMM
                model instead of setting them to be uniform across all the modes.
                Only used if @prior_use_gmm is True.

            prior_use_categorical (bool): if True, use a categorical prior instead of
                a unimodal Gaussian prior. This will also cause the encoder to output
                a categorical distribution, and will use the Gumbel-Softmax trick
                for reparametrization.

            prior_categorical_dim (int): categorical dimension - each latent sampled
                from the prior will be of shape (@latent_dim, @prior_categorical_dim)
                and will be "one-hot" in the latter dimension. Only used if
                @prior_use_categorical is True.

            prior_categorical_gumbel_softmax_hard (bool): if True, use the "hard" version of
                Gumbel Softmax for reparametrization. Only used if @prior_use_categorical is True.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MIMO_Transformer, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes]
        )
        assert isinstance(output_shapes, OrderedDict)

        # set all useful params
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.transformer_relative_timestep = transformer_relative_timestep
        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_euclidean_distance_timestep = transformer_euclidean_distance_timestep
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_max_timestep = transformer_max_timestep
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps
        self.transformer_language_enabled = transformer_language_enabled
        self.transformer_language_embedding = transformer_language_embedding
        self.transformer_finetune_language_embedding = transformer_finetune_language_embedding
        self.transformer_primitive_type = transformer_primitive_type
        self.transformer_use_cross_attention_conditioning = (
            transformer_use_cross_attention_conditioning
        )
        self.transformer_use_alternating_cross_attention_conditioning = (
            transformer_use_alternating_cross_attention_conditioning
        )
        self.transformer_add_primitive_id = transformer_add_primitive_id
        self.transformer_tokenize_primitive_id = transformer_tokenize_primitive_id
        self.transformer_channel_condition = transformer_channel_condition
        self.transformer_tokenize_obs_components = transformer_tokenize_obs_components
        self.transformer_num_patches_per_image_dim = transformer_num_patches_per_image_dim
        self.transformer_nets_to_freeze = transformer_nets_to_freeze
        self.transformer_use_ndp_decoder = transformer_use_ndp_decoder
        self.transformer_num_task_ids = transformer_num_task_ids

        self.use_cvae = use_cvae
        self.predict_signature = predict_signature
        self.replan_every_step = replan_every_step
        self.transformer_decoder = transformer_decoder
        self.prior_use_categorical = prior_use_categorical
        self.prior_categorical_gumbel_softmax_hard = prior_categorical_gumbel_softmax_hard
        self._gumbel_temperature = 1.0
        self.latent_dim = latent_dim
        self.prior_categorical_dim = prior_categorical_dim
        self.language_as_task_id = transformer_language_as_task_id
        self.obs_encoder_output_dim = transformer_embed_dim
        self.layer_dims = layer_dims
        self.prior_use_gmm = prior_use_gmm
        self.prior_gmm_num_modes = prior_gmm_num_modes
        self.prior_gmm_learn_weights = prior_gmm_learn_weights

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
            return_dict_features=transformer_tokenize_obs_components,
            num_patches_per_image_dim=transformer_num_patches_per_image_dim,
        )

        if self.latent_dim > 0:
            self.build_latent_planner(
                use_cvae, prior_use_categorical, latent_dim, prior_categorical_dim, encoder_kwargs
            )

        if transformer_num_task_ids > 1:
            self.nets["embed_task_id"] = nn.Embedding(
                transformer_num_task_ids, transformer_task_id_embed_dim
            )
            self.obs_encoder_output_dim -= transformer_task_id_embed_dim

        if transformer_primitive_type != "none":
            if (
                self.transformer_use_cross_attention_conditioning
                or self.transformer_use_alternating_cross_attention_conditioning
                or self.transformer_tokenize_primitive_id
                or self.transformer_add_primitive_id
            ):
                self.nets["embed_primitive_id"] = nn.Embedding(1000, transformer_embed_dim)
            else:
                self.nets["embed_primitive_id"] = nn.Embedding(1000, transformer_task_id_embed_dim)
                self.obs_encoder_output_dim -= transformer_task_id_embed_dim

        if self.transformer_channel_condition:
            self.nets["embed_channel_condition"] = nn.Embedding(1000, 1)

        if transformer_language_enabled:
            self.build_language_encoder(
                transformer_language_as_task_id,
                transformer_embed_dim,
                transformer_language_embedding,
            )

        if self.transformer_tokenize_obs_components:
            self.nets["embed_encoder"] = nn.ModuleDict()
            self.obs_components_keys = []
            for k, v in self.nets["encoder"].output_shape()["obs"].items():
                self.obs_components_keys.append(k)
                if k.endswith("image"):
                    for i in range(
                        self.transformer_num_patches_per_image_dim
                        * self.transformer_num_patches_per_image_dim
                    ):
                        self.nets["embed_encoder"][k + "_patch_" + str(i)] = nn.Linear(
                            v // (self.transformer_num_patches_per_image_dim**2),
                            self.obs_encoder_output_dim,
                        )
                else:
                    self.nets["embed_encoder"][k] = nn.Linear(v, self.obs_encoder_output_dim)
        else:
            input_dim = self.nets["encoder"].output_shape()[0]
            self.nets["embed_encoder"] = nn.ModuleDict()
            self.obs_components_keys = ["all"]
            self.nets["embed_encoder"]["all"] = nn.Linear(input_dim, self.obs_encoder_output_dim)

        self.build_positional_embedding_nets()

        self.transformer_context_length *= len(self.nets["embed_encoder"].keys())

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim // len(input_obs_group_shapes))

        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_embedding_dropout)

        # GPT transformer
        if transformer_type == "gpt":
            self.nets["transformer"] = GPT_Backbone(
                embed_dim=transformer_embed_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                context_length=self.transformer_context_length,
                block_attention_dropout=transformer_block_attention_dropout,
                block_output_dropout=transformer_block_output_dropout,
                use_custom_transformer_block=transformer_use_custom_transformer_block,
                activation=transformer_activation,
                block_drop_path=transformer_block_drop_path,
                use_cross_attention_conditioning=transformer_use_cross_attention_conditioning,
                use_alternating_cross_attention_conditioning=transformer_use_alternating_cross_attention_conditioning,
                key_value_from_condition=transformer_key_value_from_condition,
            )
        elif transformer_type == "mega":
            self.nets["transformer"] = MEGA_Backbone(
                embed_dim=transformer_embed_dim,
                context_length=self.transformer_context_length,
                num_layers=transformer_num_layers,
                mega_kwargs=mega_kwargs,
            )

        # transformer decoder
        if transformer_decoder:
            self.build_transformer_decoder()

        # decoder for output modalities
        if self.transformer_use_ndp_decoder:
            if transformer_ndp_decoder_kwargs is None:
                transformer_ndp_decoder_kwargs = {}
            self.nets["decoder"] = NDPObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=transformer_embed_dim,
                ndp_decoder_kwargs=transformer_ndp_decoder_kwargs,
            )
        else:
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=transformer_embed_dim,
            )
        self.freeze_networks()

    def freeze_networks(self):
        if "encoder" in self.transformer_nets_to_freeze:
            for k, net in self.nets.items():
                if "encoder" in k:
                    freeze_net(net)
                if "embed" in k:
                    freeze_net(net)

        if "decoder" in self.transformer_nets_to_freeze:
            freeze_net(self.nets["decoder"])

        if "transformer" in self.transformer_nets_to_freeze:
            freeze_net(self.nets["transformer"])

        if "attention" in self.transformer_nets_to_freeze:
            for net in list(self.nets["transformer"].nets["transformer"]):
                for k, net_ in net.nets.items():
                    if "attention" in k:
                        freeze_net(net_)

        if "mlp" in self.transformer_nets_to_freeze:
            for net in list(self.nets["transformer"].nets["transformer"]):
                for k, net_ in net.nets.items():
                    if "mlp" in k:
                        freeze_net(net_)

        if "all_but_last" in self.transformer_nets_to_freeze:
            for idx, net in enumerate(list(self.nets["transformer"].nets["transformer"])):
                if idx != len(list(self.nets["transformer"].nets["transformer"])) - 1:
                    freeze_net(net)

    def build_positional_embedding_nets(self):
        if self.transformer_relative_timestep and not self.transformer_euclidean_distance_timestep:
            max_timestep = self.transformer_context_length
        else:
            max_timestep = self.transformer_max_timestep

        if self.transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(self.transformer_embed_dim)
        else:
            if self.transformer_nn_parameter_for_timesteps:
                assert (
                    self.transformer_relative_timestep
                ), "nn.Parameter only works with relative timesteps"
                assert (
                    not self.transformer_sinusoidal_embedding
                ), "nn.Parameter only works with learned embeddings"
                self.params["embed_timestep"] = nn.Parameter(
                    torch.zeros(1, max_timestep, self.transformer_embed_dim)
                )
            else:
                self.nets["embed_timestep"] = nn.Embedding(max_timestep, self.transformer_embed_dim)
        if self.transformer_tokenize_obs_components:
            max_num_components = len(self.obs_components_keys)
            self.params["embed_components"] = nn.Parameter(
                torch.zeros(1, max_num_components, self.transformer_embed_dim)
            )
        # patchwise positional embedding
        self.params["embed_patches"] = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.transformer_num_patches_per_image_dim
                * self.transformer_num_patches_per_image_dim
                * 2,
                self.transformer_embed_dim,
            )
        )

    def build_transformer_decoder(self):
        output_dim = self.output_shapes["action"]
        self.nets["target_encoder"] = nn.Linear(output_dim, self.transformer_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.transformer_embed_dim,
            nhead=self.transformer_num_heads,
            dim_feedforward=4 * self.transformer_embed_dim,
            norm_first=True,
            batch_first=True,
            dropout=self.transformer_block_attention_dropout,
            activation=nn.GELU(),
        )
        self.nets["transformer_decoder"] = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.transformer_num_layers,
            norm=nn.LayerNorm(self.transformer_embed_dim),
        )

    def embed_timesteps(self, timesteps, embeddings):
        if self.transformer_relative_timestep:
            if self.transformer_euclidean_distance_timestep:
                original_timesteps = timesteps
                # since cumsum, can subtract first value to get relative position
                timesteps = timesteps - timesteps[:, 0].unsqueeze(-1)
                # to account for any timestep masking:
                timesteps = ((original_timesteps).abs() > 0).float() * timesteps
                timesteps = timesteps[:, :, 0]
            else:
                timesteps = (
                    torch.arange(
                        0,
                        embeddings.shape[1],
                        dtype=embeddings.dtype,
                        device=timesteps.device,
                    )
                    .unsqueeze(0)
                    .repeat(timesteps.shape[0], 1)
                )
        else:
            timesteps = timesteps[:, :, 0]
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()
            assert timesteps.max() <= self.transformer_max_timestep, timesteps.max()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            # these are NOT fed into transformer, only added to the inputs:
            time_embeddings = self.nets["embed_timestep"](timesteps)
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def update_embeddings_with_task_id(self, embeddings, task_id):
        task_id_embeddings = self.nets["embed_task_id"](task_id[:, :, 0].long())
        embeddings = {k: torch.cat([v, task_id_embeddings], dim=-1) for k, v in embeddings.items()}
        return embeddings

    def update_embeddings_with_primitive_id(self, embeddings, primitive_id):
        assert self.transformer_primitive_type != "none"
        primitive_id_embeddings = self.nets["embed_primitive_id"](primitive_id[:, :, 0].long())
        if self.transformer_add_primitive_id:
            embeddings = embeddings + primitive_id_embeddings
            primitive_id_embeddings = None
        elif self.transformer_tokenize_primitive_id:
            embeddings = torch.cat([embeddings, primitive_id_embeddings[:, 0:1, :]], dim=1)
            primitive_id_embeddings = None
        elif not (
            self.transformer_use_cross_attention_conditioning
            or self.transformer_use_alternating_cross_attention_conditioning
        ):
            embeddings = torch.cat([embeddings, primitive_id_embeddings], dim=-1)
            primitive_id_embeddings = None
        return embeddings, primitive_id_embeddings

    def update_embeddings_with_language(self, embeddings, language):
        if self.language_as_task_id:
            # language_embeddings = self.nets["embed_language"](language[:, 0, 0].long())
            # language_embeddings = language_embeddings.unsqueeze(1).repeat(1, embeddings.shape[1], 1)
            # embeddings = torch.cat([embeddings, language_embeddings], dim=-1)
            pass
        else:
            # assume the language goal does not change across an subsequence
            # language = language.long()
            language = language.long()[:, 0]
            tokens, masks = language.chunk(2, dim=-1)
            inputs = {"input_ids": tokens, "attention_mask": masks}
            # (batch_size, sentence_length, 768)
            if self.transformer_language_embedding == "raw":
                language_embeddings = self.nets["embed_language"](tokens)
            else:
                language_embeddings = self.nets["embed_language"](**inputs)[0]
            if not self.transformer_finetune_language_embedding:
                language_embeddings = language_embeddings.detach()
            embeddings = torch.cat(
                [
                    language_embeddings,
                    self.params["separator_token"].repeat((embeddings.shape[0], 1, 1)),
                    embeddings,
                ],
                dim=1,
            )
            # language_embeddings = self.nets["embed_language"](tokens)[:, :, :, 0]
            # embeddings = torch.cat([embeddings, language_embeddings], dim=-1)
        return embeddings

    def update_embeddings_with_latent_plans(self, embeddings, signature, initial_obs, latent_plan):
        signature = signature[:, 0, :]  # constant along time
        initial_obs = initial_obs[:, 0, :]  # constant along time
        inputs = OrderedDict(signature=signature)
        conditions = OrderedDict(initial_obs=initial_obs)
        if self.training or self.replan_every_step:
            cutpoint_mask = self.cutpoint_mask  
            if self.predict_signature:
                signature_pred = self.nets["predict_signature"](
                    obs=conditions,
                )["signature"]
                kl_loss = F.mse_loss(signature_pred, signature)
                latent_plan = self.nets["encode_signature"](
                    obs=OrderedDict(
                        signature=signature_pred.detach()
                    )  # detach to make it like the oracle exp
                )["latent_plan"]
            else:
                posterior_params = self.nets["encode_signature"](
                    input=inputs,
                )
                if self.prior_use_categorical:
                    # reshape to [B, D, C] to take softmax across categorical classes
                    logits = posterior_params["logit"].reshape(
                        -1, self.latent_dim, self.prior_categorical_dim
                    )
                    z = F.gumbel_softmax(
                        logits=logits,
                        tau=self._gumbel_temperature,
                        hard=self.prior_categorical_gumbel_softmax_hard,
                        dim=-1,
                    )
                    # reshape to [B, D * C], since downstream networks expect flat latents
                    latent_plan = TensorUtils.flatten(z)
                else:
                    latent_plan = TorchUtils.reparameterize(
                        mu=posterior_params["mean"],
                        logvar=posterior_params["logvar"],
                    )
                if not self.training:
                    latent_plan = (
                        self.nets["prior"]
                        .sample(embeddings.shape[0], obs_dict=conditions if self.use_cvae else None)
                        .to(latent_plan.device)
                    )
                kl_loss = self.nets["prior"].kl_loss(
                    posterior_params=posterior_params,
                    z=latent_plan,
                    obs_dict=conditions if self.use_cvae else None,
                )
            latent_plan = (
                latent_plan.unsqueeze(1).repeat(1, embeddings.shape[1], 1) * cutpoint_mask
            )  # repeat along context
        else:
            if self.predict_signature:
                signature_pred = self.nets["predict_signature"](
                    obs=conditions,
                )["signature"]
                kl_loss = F.mse_loss(signature_pred, signature)
                latent_plan = self.nets["encode_signature"](
                    obs=OrderedDict(
                        signature=signature_pred.detach()
                    )  # detach to make it like the oracle exp
                )["latent_plan"]
                latent_plan = latent_plan.unsqueeze(1).repeat(
                    1, embeddings.shape[1], 1
                )  # repeat along context
            else:
                kl_loss = 0  # dummy its not used during training time
        embeddings = torch.cat([embeddings, latent_plan], dim=-1)
        return embeddings, kl_loss

    def generate_channel_condition_img(self, img, condition):
        channel_img = torch.ones_like(img, device=img.device)[:, :, 0:1, :, :]
        condition = self.nets["embed_channel_condition"](condition.long()[:, :, 0])
        condition = condition.unsqueeze(-1).unsqueeze(-1)
        channel_img = channel_img * condition
        return channel_img

    def extract_from_obs(self, inputs):
        # do not encode timesteps through encoder network!
        timesteps = inputs["obs"]["timesteps"]
        del inputs["obs"]["timesteps"]

        if "task_id" in inputs["obs"] and self.transformer_num_task_ids > 1:
            # do not encode task_id through encoder network!
            task_id = inputs["obs"]["task_id"]
            del inputs["obs"]["task_id"]
        else:
            task_id = None

        if "latent_plan" in inputs["obs"]:
            # do not encode latent_plan through encoder network!
            latent_plan = inputs["obs"]["latent_plan"]
            del inputs["obs"]["latent_plan"]
        else:
            latent_plan = None

        if "action" in inputs["obs"]:
            # this is for the decoder to get the actions
            # do not encode actions through encoder network!
            actions = inputs["obs"]["action"]
            del inputs["obs"]["action"]
        else:
            actions = None

        if "transformer_encoder_outputs" in inputs["obs"]:
            # do not encode encoder outputs through encoder network!
            transformer_encoder_outputs = inputs["obs"]["transformer_encoder_outputs"]
            del inputs["obs"]["transformer_encoder_outputs"]
        else:
            transformer_encoder_outputs = None

        if (
            "signature" in inputs["obs"]
            and "initial_obs" in inputs["obs"]
            and self.kl_loss_weight > 0
        ):
            # do not encode signature through encoder network!
            signature = inputs["obs"]["signature"]
            del inputs["obs"]["signature"]
            # do not encode initial_obs through encoder network!
            initial_obs = inputs["obs"]["initial_obs"]
            del inputs["obs"]["initial_obs"]
        else:
            signature = None
            initial_obs = None

        if self.transformer_language_enabled:
            # do not encode language through encoder network!
            language = inputs["obs"]["language"]
            del inputs["obs"]["language"]
            if not self.language_as_task_id:
                new_obs = {}
                for k, v in inputs["obs"].items():
                    new_obs[k] = v[:, language.shape[-1] // 2 + 1 :, :]
                inputs["obs"] = new_obs
            if "agentview_image" in inputs["obs"]:
                channel_img = self.generate_channel_condition_img(
                    inputs["obs"]["agentview_image"], language
                )
                inputs["obs"]["agentview_image"] = torch.cat(
                    [inputs["obs"]["agentview_image"], channel_img], dim=2
                )
                inputs["obs"]["robot0_eye_in_hand_image"] = torch.cat(
                    [inputs["obs"]["robot0_eye_in_hand_image"], channel_img], dim=2
                )
                language = None
            # else:
            #     new_obs = {}
            #     for k, v in inputs["obs"].items():
            #         new_obs[k] = v[:, 1:, :]
            #     inputs["obs"] = new_obs
        else:
            language = None
        if "combinatorial_stack_id" in inputs["obs"] and self.transformer_channel_condition:
            # do not encode combinatorial_stack_id through encoder network!
            combinatorial_stack_id = inputs["obs"]["combinatorial_stack_id"]
            del inputs["obs"]["combinatorial_stack_id"]
            if "agentview_image" in inputs["obs"]:
                channel_img = self.generate_channel_condition_img(
                    inputs["obs"]["agentview_image"], combinatorial_stack_id
                )
                inputs["obs"]["agentview_image"] = torch.cat(
                    [inputs["obs"]["agentview_image"], channel_img], dim=2
                )
                inputs["obs"]["robot0_eye_in_hand_image"] = torch.cat(
                    [inputs["obs"]["robot0_eye_in_hand_image"], channel_img], dim=2
                )
                combinatorial_stack_id = None
        else:
            if "combinatorial_stack_id" in inputs["obs"]:
                combinatorial_stack_id = inputs["obs"]["combinatorial_stack_id"]
                del inputs["obs"]["combinatorial_stack_id"]
            else:
                combinatorial_stack_id = None

        if self.transformer_primitive_type == "combinatorial_stack_id":
            primitive_id = combinatorial_stack_id
        else:
            primitive_id = None

        if self.transformer_tokenize_primitive_id:
            new_obs = {}
            for k, v in inputs["obs"].items():
                new_obs[k] = v[:, 1:, :]
            inputs["obs"] = new_obs

        return (
            inputs,
            timesteps,
            task_id,
            latent_plan,
            actions,
            transformer_encoder_outputs,
            signature,
            initial_obs,
            language,
            primitive_id,
        )

    def input_embedding(
        self,
        inputs,
        timesteps,
        task_id=None,
        language=None,
        signature=None,
        initial_obs=None,
        latent_plan=None,
        primitive_id=None,
    ):
        if type(inputs) is not dict:
            inputs = {"obs": {"all": inputs}}
        embeddings = {
            k: self.nets["embed_encoder"][k](inputs["obs"][k]) for k in self.obs_components_keys
        }
        if task_id is not None:
            embeddings = self.update_embeddings_with_task_id(embeddings, task_id)

        if language is not None:
            embeddings = self.update_embeddings_with_language(embeddings, language)

        if primitive_id is not None:
            embeddings, primitive_id_embeddings = self.update_embeddings_with_primitive_id(
                embeddings, primitive_id
            )
        else:
            primitive_id_embeddings = None

        if self.kl_loss_weight > 0 or self.predict_signature:
            embeddings, kl_loss = self.update_embeddings_with_latent_plans(
                embeddings, signature, initial_obs, latent_plan
            )
        else:
            kl_loss = embeddings[list(embeddings.keys())[0]][0, 0, 0] * 0  # dummy kl loss

        new_embeddings = OrderedDict()
        for idx, k in enumerate(self.obs_components_keys):
            v = embeddings[k]
            time_embeddings = self.embed_timesteps(timesteps, v)
            v = v + time_embeddings
            if self.transformer_tokenize_obs_components:
                v += self.params["embed_components"][:, idx]
                # patch embeddings 0-15 are for agentview
                # patch embeddings 16-31 are for eye_in_hand
                if "eye_in_hand_image" in k:
                    p = int(k.split("_")[-1])
                    p += (
                        self.transformer_num_patches_per_image_dim
                        * self.transformer_num_patches_per_image_dim
                    )
                    v = v + self.params["embed_patches"][:, :, p]
                elif "agentview_image" in k:
                    p = int(k.split("_")[-1])
                    v = v + self.params["embed_patches"][:, :, 1]
            v = self.nets["embed_ln"](v)
            v = self.nets["embed_drop"](v)
            new_embeddings[k] = v

        # embeddings are K k:v pairs with v having shapes BxTxD
        # need to cat them so that we have a single tensor of shape Bx(T*K)xD
        # this is valid because we have provided timestep embeddings to each of the K tensors
        embeddings = torch.cat([v for k, v in new_embeddings.items()], dim=1)
        return embeddings, kl_loss, primitive_id_embeddings

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def get_outputs_from_encoder(self, transformer_encoder_outputs, actions, timesteps):
        if self.transformer_decoder:
            tgt_mask = self._generate_square_subsequent_mask(actions.shape[1], actions.shape[1]).to(
                transformer_encoder_outputs.device
            )
            memory_mask = self._generate_square_subsequent_mask(
                self.transformer_context_length, actions.shape[1]
            ).to(transformer_encoder_outputs.device)

            embeddings = self.nets["target_encoder"](actions)
            time_embeddings = self.embed_timesteps(timesteps, embeddings)[:, : actions.shape[1]]
            embeddings = embeddings + time_embeddings
            transformer_outputs = self.nets["transformer_decoder"].forward(
                tgt=embeddings,
                memory=transformer_encoder_outputs,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        else:
            transformer_outputs = transformer_encoder_outputs
        return transformer_outputs

    def forward(self, inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )
        inputs = inputs.copy()

        (
            inputs,
            timesteps,
            task_id,
            latent_plan,
            actions,
            transformer_encoder_outputs,
            signature,
            initial_obs,
            language,
            command_index,
        ) = self.extract_from_obs(inputs)

        # use encoder to each timestep of sequence to extract flat transformer inputs
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        # assert transformer_inputs.ndim == 3  # [B, T, D]

        if transformer_encoder_outputs is None:
            transformer_embeddings, kl_loss, condition = self.input_embedding(
                transformer_inputs,
                timesteps,
                task_id,
                language,
                signature,
                initial_obs,
                latent_plan,
                command_index,
            )
            # make condition shape match transformer_embeddings
            if condition is not None:
                condition = condition.repeat(
                    1, transformer_embeddings.shape[1] // condition.shape[1], 1
                )
                assert transformer_embeddings.shape == condition.shape
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(
                transformer_embeddings, condition
            )
        else:
            kl_loss = 0

        transformer_outputs = self.get_outputs_from_encoder(
            transformer_encoder_outputs, actions, timesteps
        )

        # apply decoder to each timestep of sequence to get a dictionary of outputs
        if self.transformer_use_ndp_decoder:
            transformer_outputs = TensorUtils.time_distributed(
                OrderedDict(
                    feats=transformer_outputs,
                    proprio=torch.cat(
                        (inputs["obs"]["joint_pos"], inputs["obs"]["gripper_qpos"]), dim=-1
                    ),
                ),
                self.nets["decoder"],
            )
        else:
            transformer_outputs = TensorUtils.time_distributed(
                transformer_outputs, self.nets["decoder"]
            )
        transformer_outputs["kl_loss"] = kl_loss
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def build_latent_planner(self, input_obs_group_shapes, encoder_kwargs):
        encoder_obs_group_shapes = OrderedDict()
        encoder_obs_group_shapes["input"] = OrderedDict(OrderedDict(signature=[399]))
        prior_obs_group_shapes = OrderedDict(condition=None)
        if self.use_cvae:
            prior_learnable = True
            init_obs_shape = sum(
                input_obs_group_shapes["obs"][k][0] for k in input_obs_group_shapes["obs"]
            )
            prior_obs_group_shapes["condition"] = OrderedDict(
                OrderedDict(initial_obs=[init_obs_shape])
            )
        else:
            prior_learnable = False

        if self.prior_use_categorical:
            encoder_output_shapes = OrderedDict(
                logit=(self.latent_dim * self.prior_categorical_dim,),
            )
        else:
            encoder_output_shapes = OrderedDict(
                mean=(self.latent_dim,),
                logvar=(self.latent_dim,),
            )

        self.nets["encode_signature"] = MIMO_MLP(
            input_obs_group_shapes=encoder_obs_group_shapes,
            output_shapes=encoder_output_shapes,
            layer_dims=self.layer_dims,
            encoder_kwargs=encoder_kwargs,
        )
        if self.prior_use_categorical:
            self.nets["prior"] = CategoricalPrior(
                latent_dim=self.latent_dim,
                categorical_dim=self.prior_categorical_dim,
                device=torch.device("cuda"),  # this won't work once we use multi-GPU
                learnable=prior_learnable,
                obs_shapes=prior_obs_group_shapes["condition"],
                mlp_layer_dims=self.layer_dims,
                encoder_kwargs=encoder_kwargs,
            )
            self.obs_encoder_output_dim -= self.latent_dim * self.prior_categorical_dim
        else:
            self.nets["prior"] = GaussianPrior(
                latent_dim=self.latent_dim,
                device=torch.device("cuda"),  # this won't work once we use multi-GPU
                learnable=prior_learnable,
                obs_shapes=prior_obs_group_shapes["condition"],
                use_gmm=self.prior_use_gmm,
                gmm_num_modes=self.prior_gmm_num_modes,
                gmm_learn_weights=self.prior_gmm_learn_weights,
                mlp_layer_dims=self.layer_dims,
                encoder_kwargs=encoder_kwargs,
            )
            self.obs_encoder_output_dim -= self.latent_dim

        if self.predict_signature:
            init_obs_shape = sum(
                input_obs_group_shapes["obs"][k][0] for k in input_obs_group_shapes["obs"]
            )
            self.nets["predict_signature"] = MIMO_MLP(
                input_obs_group_shapes=OrderedDict(obs=OrderedDict(initial_obs=[init_obs_shape])),
                output_shapes=OrderedDict(
                    signature=[399],
                ),
                layer_dims=self.layer_dims,
                encoder_kwargs=encoder_kwargs,
            )

            self.nets["encode_signature"] = MIMO_MLP(
                input_obs_group_shapes=OrderedDict(
                    obs=OrderedDict(
                        signature=[399],
                    )
                ),
                output_shapes=OrderedDict(
                    latent_plan=[self.latent_dim],
                ),
                layer_dims=self.layer_dims,
                encoder_kwargs=encoder_kwargs,
            )

    def build_language_encoder(
        self, language_as_task_id, transformer_embed_dim, transformer_language_embedding
    ):
        self.nets["embed_language"] = nn.Embedding(4, 1)

    def _generate_square_subsequent_mask(self, sz1, sz2):
        # https://github.com/pytorch/examples/blob/main/word_language_model/model.py
        mask = (torch.triu(torch.ones(sz1, sz2)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer={}".format(self.nets["transformer"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class TransformerActorNetwork(MIMO_Transformer):
    """
    An Transformer policy network that predicts actions from observation sequences (assumed to be frame stacked
    from previous observations) and possible from previous actions as well (in an autoregressive manner).
    """

    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_embedding_dropout=0.1,
        transformer_block_attention_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_block_drop_path=0.0,
        transformer_condition_on_actions=False,
        transformer_predict_obs=False,
        transformer_relative_timestep=False,
        transformer_max_timestep=1250,
        transformer_euclidean_distance_timestep=False,
        transformer_sinusoidal_embedding=False,
        transformer_use_custom_transformer_block=True,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        transformer_task_id_embed_dim=0,
        transformer_num_task_ids=1,
        transformer_language_enabled=False,
        transformer_language_embedding="raw",
        transformer_finetune_language_embedding=False,
        transformer_decoder=False,
        transformer_primitive_type="none",
        transformer_use_cross_attention_conditioning=False,
        transformer_use_alternating_cross_attention_conditioning=False,
        transformer_key_value_from_condition=False,
        transformer_add_primitive_id=False,
        transformer_tokenize_primitive_id=False,
        transformer_channel_condition=False,
        transformer_tokenize_obs_components=False,
        transformer_num_patches_per_image_dim=1,
        transformer_nets_to_freeze=(),
        transformer_use_ndp_decoder=False,
        transformer_ndp_decoder_kwargs=None,
        transformer_type="gpt",
        mega_kwargs=None,
        use_cvae=True,
        predict_signature=False,
        layer_dims=(1024, 1024),
        latent_dim=16,
        prior_use_gmm=True,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=True,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        replan_every_step=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block

            transformer_condition_on_actions (bool): if True, transformer will condition on the sequence of actions
                in addition to the observation sequence. In this case, the inputs to the policy are expected to
                contain an "action" key in the observation dictionaries.

            transformer_predict_obs (bool): if True, transformer will predict observations in the output sequences
                as well. In this case, the policy outputs dictionaries containing sequences, instead of raw
                action sequences.

            transformer_relative_timestep (bool): if True, timesteps range from 0 to context length - 1, if False use absolute position in trajectory.

            transformer_max_timestep (int): when using absolute timesteps or euclidean distance timesteps,
                define the maximal timestep value

            transformer_euclidean_distance_timestep (int): if True, use cumulative end-effector distance traveled as timesteps

            transformer_sinusoidal_embedding (bool): if True, use sinusoidal positional embeddings that are not learned

            transformer_use_custom_transformer_block (bool): if True, use custom transformer block

            transformer_activation (str): string denoting the activation function to use in each transformer block

            transformer_nn_parameter_for_timesteps (bool): if True, use nn.Parameter for embedding timesteps

            transformer_task_id_embed_dim (int): use nn.Embedding to embed task ids

            transformer_num_task_ids (int): number of tasks we are training with

            transformer_language_enabled (bool): if True, condition on language embeddings

            transformer_language_embedding (str): string denoting the language embedding to use

            transformer_finetune_language_embedding (bool): if True, finetune the language embedding

            transformer_use_cross_attention_conditioning (bool): if True, use cross attention conditioning on the input sequence

            transformer_use_alternating_cross_attention_conditioning (bool): if True, use cross attention conditioning on the input sequence

            transformer_tokenize_obs_components (bool): if True, tokenize the observation components

            use_cvae (bool): if True, use condition on initial obs for the prior and encoder

            predict_signature (bool): if True, instead of VIB, use latents from the encoder to predict the signature
            and condition transformer

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

            latent_dim (int): dimension of latent space for the VAE

            prior_use_gmm (bool): if True, learn a Gaussian Mixture Model (GMM)
                prior instead of a unimodal Gaussian prior.

            prior_gmm_num_modes (int): number of GMM modes to learn. Only
                used if @prior_use_gmm is True.

            prior_gmm_learn_weights (bool): if True, learn the weights of the GMM
                model instead of setting them to be uniform across all the modes.
                Only used if @prior_use_gmm is True.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.ac_dim = ac_dim

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        self.transformer_condition_on_actions = transformer_condition_on_actions
        self.transformer_predict_obs = transformer_predict_obs
        self.transformer_use_custom_transformer_block = transformer_use_custom_transformer_block
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps
        self.transformer_decoder = transformer_decoder

        # set up different observation groups for @Transformer_MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        del observation_group_shapes["obs"]["timesteps"]
        if self.transformer_condition_on_actions:
            # add actions since policy will condition on and predict actions in addition to observations
            observation_group_shapes["obs"]["action"] = (ac_dim,)

        if "task_id" in observation_group_shapes["obs"]:
            del observation_group_shapes["obs"]["task_id"]

        if "language" in observation_group_shapes["obs"]:
            del observation_group_shapes["obs"]["language"]
            if "agentview_image" in observation_group_shapes["obs"]:
                observation_group_shapes["obs"]["agentview_image"][0] += 1
            if "robot0_eye_in_hand_image" in observation_group_shapes["obs"]:
                observation_group_shapes["obs"]["robot0_eye_in_hand_image"][0] += 1

        if "combinatorial_stack_id" in observation_group_shapes["obs"]:
            # add a channel for the combinatorial stack id
            if transformer_channel_condition:
                if "agentview_image" in observation_group_shapes["obs"] and not (
                    transformer_language_enabled
                ):
                    observation_group_shapes["obs"]["agentview_image"][0] += 1
                if "robot0_eye_in_hand_image" in observation_group_shapes["obs"] and not (
                    transformer_language_enabled
                ):
                    observation_group_shapes["obs"]["robot0_eye_in_hand_image"][0] += 1
            del observation_group_shapes["obs"]["combinatorial_stack_id"]

        if (
            "signature" in observation_group_shapes["obs"]
            and "initial_obs" in observation_group_shapes["obs"]
        ):
            del observation_group_shapes["obs"]["signature"]
            del observation_group_shapes["obs"]["initial_obs"]

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(TransformerActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_embedding_dropout=transformer_embedding_dropout,
            transformer_block_attention_dropout=transformer_block_attention_dropout,
            transformer_block_output_dropout=transformer_block_output_dropout,
            transformer_block_drop_path=transformer_block_drop_path,
            transformer_relative_timestep=transformer_relative_timestep,
            transformer_max_timestep=transformer_max_timestep,
            transformer_euclidean_distance_timestep=transformer_euclidean_distance_timestep,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_use_custom_transformer_block=transformer_use_custom_transformer_block,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            transformer_task_id_embed_dim=transformer_task_id_embed_dim,
            transformer_num_task_ids=transformer_num_task_ids,
            transformer_language_enabled=transformer_language_enabled,
            transformer_language_embedding=transformer_language_embedding,
            transformer_finetune_language_embedding=transformer_finetune_language_embedding,
            transformer_decoder=transformer_decoder,
            transformer_primitive_type=transformer_primitive_type,
            transformer_use_cross_attention_conditioning=transformer_use_cross_attention_conditioning,
            transformer_use_alternating_cross_attention_conditioning=transformer_use_alternating_cross_attention_conditioning,
            transformer_key_value_from_condition=transformer_key_value_from_condition,
            transformer_add_primitive_id=transformer_add_primitive_id,
            transformer_tokenize_primitive_id=transformer_tokenize_primitive_id,
            transformer_channel_condition=transformer_channel_condition,
            transformer_tokenize_obs_components=transformer_tokenize_obs_components,
            transformer_num_patches_per_image_dim=transformer_num_patches_per_image_dim,
            transformer_nets_to_freeze=transformer_nets_to_freeze,
            transformer_use_ndp_decoder=transformer_use_ndp_decoder,
            transformer_ndp_decoder_kwargs=transformer_ndp_decoder_kwargs,
            transformer_type=transformer_type,
            mega_kwargs=mega_kwargs,
            use_cvae=use_cvae,
            predict_signature=predict_signature,
            layer_dims=layer_dims,
            latent_dim=latent_dim,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            replan_every_step=replan_every_step,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_Transformer, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        output_shapes = OrderedDict(action=(self.ac_dim,))
        if self.transformer_predict_obs:
            # predict observations too
            for k in self.obs_shapes:
                if k != "timesteps":
                    output_shapes[k] = self.obs_shapes[k]
        return output_shapes

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        output_shapes = {k: list(self.output_shapes[k]) for k in self.output_shapes}
        if self.transformer_predict_obs:
            return output_shapes
        else:
            return output_shapes["action"]

    def forward(self, obs_dict, actions=None, goal_dict=None):
        """
        Forward a sequence of inputs through the Transformer.

        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            actions (torch.Tensor): batch of actions of shape [B, T, D]. Only required
                if @self.transformer_condition_on_actions is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            outputs (torch.Tensor or dict): contains predicted action sequence, or dictionary
                with predicted action sequence and predicted observation sequences depending
                on @self.transformer_predict_obs
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        if self.transformer_condition_on_actions or self.transformer_decoder:
            forward_kwargs["obs"]["action"] = actions
        outputs = super(TransformerActorNetwork, self).forward(forward_kwargs)

        # apply tanh squashing to ensure actions are in [-1, 1]
        outputs["action"] = torch.tanh(outputs["action"])

        if self.transformer_predict_obs:
            return outputs  # full dictionary of output sequences including obs and action
        return outputs["action"]  # only action sequences

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)

    def configure_optimizers(self, lr, betas, weight_decay):
        """
        Function and docstring (mostly) from Andrej.

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        Args:
            lr (float): learning rate for AdamW optimizer
            betas (2-tuple): betas used by AdamW optimizer
            weight_decay (float): strength of L2 regularization
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        if self.transformer_use_custom_transformer_block:
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, torch.nn.parameter.Parameter)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module is not decayed
            if self.transformer_nn_parameter_for_timesteps:
                no_decay.add("params.embed_timestep")

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert (
                len(inter_params) == 0
            ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert (
                len(param_dict.keys() - union_params) == 0
            ), "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),
            )

            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)
        return optimizer


class TransformerGMMActorNetwork(TransformerActorNetwork):
    """
    A Transformer GMM policy network that predicts sequences of action distributions from observation
    sequences (assumed to be frame stacked from previous observations).
    """

    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_embedding_dropout=0.1,
        transformer_block_attention_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_block_drop_path=0.0,
        transformer_condition_on_actions=False,
        transformer_predict_obs=False,
        transformer_relative_timestep=False,
        transformer_max_timestep=1250,
        transformer_euclidean_distance_timestep=False,
        transformer_sinusoidal_embedding=False,
        transformer_use_custom_transformer_block=True,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        transformer_task_id_embed_dim=0,
        transformer_num_task_ids=1,
        transformer_language_enabled=False,
        transformer_language_embedding="raw",
        transformer_finetune_language_embedding=False,
        transformer_decoder=False,
        transformer_primitive_type="none",
        transformer_use_cross_attention_conditioning=False,
        transformer_use_alternating_cross_attention_conditioning=False,
        transformer_key_value_from_condition=False,
        transformer_add_primitive_id=False,
        transformer_tokenize_primitive_id=False,
        transformer_channel_condition=False,
        transformer_tokenize_obs_components=False,
        transformer_num_patches_per_image_dim=1,
        transformer_nets_to_freeze=(),
        transformer_use_ndp_decoder=False,
        transformer_ndp_decoder_kwargs=None,
        transformer_type="gpt",
        mega_kwargs=None,
        use_cvae=True,
        predict_signature=False,
        layer_dims=(1024, 1024),
        latent_dim=16,
        prior_use_gmm=True,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=True,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        replan_every_step=False,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block

            transformer_condition_on_actions (bool): if True, transformer will condition on the sequence of actions
                in addition to the observation sequence. In this case, the inputs to the policy are expected to
                contain an "action" key in the observation dictionaries.

            transformer_predict_obs (bool): not supported (yet)

            transformer_relative_timestep (bool): if True, timesteps range from 0 to context length - 1, if False use absolute position in trajectory.

            transformer_max_timestep (int): when using absolute timesteps or euclidean distance timesteps,
                define the maximal timestep value

            transformer_euclidean_distance_timestep (int): if True, use cumulative end-effector distance traveled as timesteps

            transformer_sinusoidal_embedding (bool): if True, use sinusoidal positional embeddings that are not learned

            transformer_use_custom_transformer_block (bool): if True, use custom transformer block

            transformer_activation (str): string denoting the activation function to use in each transformer block

            transformer_nn_parameter_for_timesteps (bool): if True, use nn.Parameter for embedding timesteps

            transformer_task_id_embed_dim (int): use nn.Embedding to embed task ids

            transformer_num_task_ids (int): number of tasks we are training with

            transformer_language_enabled (bool): if True, condition on language embeddings

            transformer_language_embedding (str): string denoting the language embedding to use

            transformer_finetune_language_embedding (bool): if True, finetune the language embedding

            transfomer_tokenize_obs_components (bool): if True, tokenize the observation components

            use_cvae (bool): if True, use condition on initial obs for the prior and encoder

            predict_signature (bool): if True, instead of VIB, use latents from the encoder to predict the signature
            and condition transformer

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

            latent_dim (int): dimension of latent space for the VAE

            prior_use_gmm (bool): if True, learn a Gaussian Mixture Model (GMM)
                prior instead of a unimodal Gaussian prior.

            prior_gmm_num_modes (int): number of GMM modes to learn. Only
                used if @prior_use_gmm is True.

            prior_gmm_learn_weights (bool): if True, learn the weights of the GMM
                model instead of setting them to be uniform across all the modes.
                Only used if @prior_use_gmm is True.

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh
        self.transformer_use_custom_transformer_block = transformer_use_custom_transformer_block

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert (
            std_activation in self.activations
        ), "std_activation must be one of: {}; instead got: {}".format(
            self.activations.keys(), std_activation
        )
        self.std_activation = std_activation

        super(TransformerGMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_embedding_dropout=transformer_embedding_dropout,
            transformer_block_attention_dropout=transformer_block_attention_dropout,
            transformer_block_output_dropout=transformer_block_output_dropout,
            transformer_block_drop_path=transformer_block_drop_path,
            transformer_condition_on_actions=transformer_condition_on_actions,
            transformer_predict_obs=transformer_predict_obs,
            transformer_relative_timestep=transformer_relative_timestep,
            transformer_max_timestep=transformer_max_timestep,
            transformer_euclidean_distance_timestep=transformer_euclidean_distance_timestep,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_use_custom_transformer_block=transformer_use_custom_transformer_block,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            transformer_task_id_embed_dim=transformer_task_id_embed_dim,
            transformer_num_task_ids=transformer_num_task_ids,
            transformer_language_enabled=transformer_language_enabled,
            transformer_language_embedding=transformer_language_embedding,
            transformer_finetune_language_embedding=transformer_finetune_language_embedding,
            transformer_decoder=transformer_decoder,
            transformer_primitive_type=transformer_primitive_type,
            transformer_use_cross_attention_conditioning=transformer_use_cross_attention_conditioning,
            transformer_use_alternating_cross_attention_conditioning=transformer_use_alternating_cross_attention_conditioning,
            transformer_key_value_from_condition=transformer_key_value_from_condition,
            transformer_add_primitive_id=transformer_add_primitive_id,
            transformer_tokenize_primitive_id=transformer_tokenize_primitive_id,
            transformer_channel_condition=transformer_channel_condition,
            transformer_tokenize_obs_components=transformer_tokenize_obs_components,
            transformer_num_patches_per_image_dim=transformer_num_patches_per_image_dim,
            transformer_nets_to_freeze=transformer_nets_to_freeze,
            transformer_use_ndp_decoder=transformer_use_ndp_decoder,
            transformer_ndp_decoder_kwargs=transformer_ndp_decoder_kwargs,
            transformer_type=transformer_type,
            mega_kwargs=mega_kwargs,
            use_cvae=use_cvae,
            predict_signature=predict_signature,
            layer_dims=layer_dims,
            latent_dim=latent_dim,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            replan_every_step=replan_every_step,
            encoder_kwargs=encoder_kwargs,
            goal_shapes=goal_shapes,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_Transformer superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        output_shapes = OrderedDict(
            mean=(self.num_modes, self.ac_dim),
            scale=(self.num_modes, self.ac_dim),
            logits=(self.num_modes,),
        )
        if self.transformer_predict_obs:
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    output_shapes[k + "_mean"] = (self.num_modes, v[0])
                    output_shapes[k + "_scale"] = (self.num_modes, v[0])
                    output_shapes[k + "_logits"] = (self.num_modes,)
        return output_shapes

    def build_dist(self, means, scales, logits, use_tanh):
        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(
            component_distribution, 1
        )  # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if use_tanh:
            # Wrap distribution with Tanh
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.0)
        return dists

    def forward_train(self, obs_dict, actions=None, goal_dict=None):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions or self.transformer_predit_obs is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        if self.transformer_condition_on_actions or self.transformer_decoder:
            forward_kwargs["obs"]["action"] = actions

        outputs = MIMO_Transformer.forward(self, forward_kwargs)

        dists = self.build_dist(
            outputs["mean"], outputs["scale"], outputs["logits"], use_tanh=self.use_tanh
        )

        if self.transformer_predict_obs:
            dists = {"action": dists}
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    dists[k] = self.build_dist(
                        outputs[k + "_mean"],
                        outputs[k + "_scale"],
                        outputs[k + "_logits"],
                        use_tanh=False,
                    )
        return dists, outputs["kl_loss"], outputs["transformer_encoder_outputs"]

    def forward(self, obs_dict, actions=None, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out, _, _ = self.forward_train(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
        if self.transformer_predict_obs:
            return {"action": out["action"].sample()}
        else:
            return out.sample()

    def _to_string(self):
        """Info to pretty print."""
        msg = (
            "action_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
                self.ac_dim,
                self.std_activation,
                self.low_noise_eval,
                self.num_modes,
                self.min_std,
            )
        )
        return msg


class TransformerDMLActorNetwork(TransformerActorNetwork):
    """
    A Transformer DML policy network that predicts sequences of action distributions from observation
    sequences (assumed to be frame stacked from previous observations).
    """

    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_embedding_dropout=0.1,
        transformer_block_attention_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_condition_on_actions=False,
        transformer_predict_obs=False,
        transformer_relative_timestep=False,
        transformer_max_timestep=1250,
        transformer_euclidean_distance_timestep=False,
        transformer_sinusoidal_embedding=False,
        transformer_use_custom_transformer_block=True,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        transformer_task_id_embed_dim=0,
        transformer_num_task_ids=1,
        transformer_language_enabled=False,
        transformer_language_embedding="raw",
        transformer_finetune_language_embedding=False,
        num_modes=2,
        num_classes=256,
        log_scale_min=-7.0,
        constant_variance=True,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block

            transformer_condition_on_actions (bool): if True, transformer will condition on the sequence of actions
                in addition to the observation sequence. In this case, the inputs to the policy are expected to
                contain an "action" key in the observation dictionaries.

            transformer_predict_obs (bool): not supported (yet)

            transformer_relative_timestep (bool): if True, timesteps range from 0 to context length - 1, if False use absolute position in trajectory.

            transformer_max_timestep (int): when using absolute timesteps or euclidean distance timesteps,
                define the maximal timestep value

            transformer_euclidean_distance_timestep (int): if True, use cumulative end-effector distance traveled as timesteps

            transformer_sinusoidal_embedding (bool): if True, use sinusoidal positional embeddings that are not learned

            transformer_use_custom_transformer_block (bool): if True, use custom transformer block

            transformer_activation (str): string denoting the activation function to use in each transformer block

            transformer_nn_parameter_for_timesteps (bool): if True, use nn.Parameter for embedding timesteps

            transformer_task_id_embed_dim (int): use nn.Embedding to embed task ids

            transformer_num_task_ids (int): number of tasks we are training with

            transformer_language_enabled (bool): if True, condition on language embeddings

            transformer_language_embedding (str): string denoting the language embedding to use

            transformer_finetune_language_embedding (bool): if True, finetune the language embedding

            num_modes (int): number of DML modes

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to DML actor
        self.num_modes = num_modes
        self.num_classes = num_classes
        self.log_scale_min = log_scale_min
        self.constant_variance = constant_variance

        self.transformer_use_custom_transformer_block = transformer_use_custom_transformer_block

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        super(TransformerDMLActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_embedding_dropout=transformer_embedding_dropout,
            transformer_block_attention_dropout=transformer_block_attention_dropout,
            transformer_block_output_dropout=transformer_block_output_dropout,
            transformer_condition_on_actions=transformer_condition_on_actions,
            transformer_predict_obs=transformer_predict_obs,
            transformer_relative_timestep=transformer_relative_timestep,
            transformer_max_timestep=transformer_max_timestep,
            transformer_euclidean_distance_timestep=transformer_euclidean_distance_timestep,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_use_custom_transformer_block=transformer_use_custom_transformer_block,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            transformer_task_id_embed_dim=transformer_task_id_embed_dim,
            transformer_num_task_ids=transformer_num_task_ids,
            transformer_language_enabled=transformer_language_enabled,
            transformer_language_embedding=transformer_language_embedding,
            transformer_finetune_language_embedding=transformer_finetune_language_embedding,
            encoder_kwargs=encoder_kwargs,
            goal_shapes=goal_shapes,
        )

        if constant_variance:
            scale = torch.randn((ac_dim), dtype=torch.float32) / np.sqrt(ac_dim)
            self.params["scale"] = nn.Parameter(scale, requires_grad=True)

    def _get_output_shapes(self):
        """
        Tells @MIMO_Transformer superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        output_shapes = OrderedDict(
            mean=(self.num_modes, self.ac_dim),
            logits=(self.num_modes,),
        )
        if not self.constant_variance:
            output_shapes["scale"] = (self.num_modes, self.ac_dim)

        if self.transformer_predict_obs:
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    output_shapes[k + "_mean"] = (self.num_modes, v[0])
                    output_shapes[k + "_scale"] = (self.num_modes, v[0])
                    output_shapes[k + "_logits"] = (self.num_modes,)
        return output_shapes

    def build_dist(self, means, scales, logits):
        if self.constant_variance:
            scales = scales.reshape((1, 1, 1, -1)).expand_as(means)
        dists = DiscreteMixLogistic(
            mean=means.permute(0, 1, 3, 2),
            log_scale=scales.permute(0, 1, 3, 2),
            logit_probs=logits.unsqueeze(2),
            num_classes=self.num_classes,
            log_scale_min=self.log_scale_min,
        )
        return dists

    def forward_train(self, obs_dict, actions=None, goal_dict=None):
        """
        Return full DML distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions or self.transformer_predit_obs is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        if self.transformer_condition_on_actions:
            forward_kwargs["obs"]["action"] = actions

        outputs = MIMO_Transformer.forward(self, forward_kwargs)

        if self.constant_variance:
            scale = self.params["scale"]
        else:
            scale = outputs["scale"]
        dists = self.build_dist(outputs["mean"], scale, outputs["logits"])

        if self.transformer_predict_obs:
            dists = {"action": dists}
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    dists[k] = self.build_dist(
                        outputs[k + "_mean"],
                        outputs[k + "_scale"],
                        outputs[k + "_logits"],
                        use_tanh=False,
                    )
        return dists

    def forward(self, obs_dict, actions=None, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
        if self.transformer_predict_obs:
            return {"action": out["action"].mean}
        else:
            return out.mean

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}, num_modes={}, num_classes={}, log_scale_min={}, constant_variance={}".format(
            self.ac_dim,
            self.num_modes,
            self.num_classes,
            self.log_scale_min,
            self.constant_variance,
        )
        return msg


class TransformerCategoricalActorNetwork(TransformerActorNetwork):
    """
    A Transformer DML policy network that predicts sequences of action distributions from observation
    sequences (assumed to be frame stacked from previous observations).
    """

    def __init__(
        self,
        obs_shapes,
        ac_dim,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_embedding_dropout=0.1,
        transformer_block_attention_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_condition_on_actions=False,
        transformer_predict_obs=False,
        transformer_relative_timestep=False,
        transformer_max_timestep=1250,
        transformer_euclidean_distance_timestep=False,
        transformer_sinusoidal_embedding=False,
        transformer_use_custom_transformer_block=True,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        transformer_task_id_embed_dim=0,
        transformer_num_task_ids=1,
        transformer_language_enabled=False,
        transformer_language_embedding="raw",
        transformer_finetune_language_embedding=False,
        num_classes=256,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            transformer_embed_dim (int): dimension for embeddings used by transformer

            transformer_num_layers (int): number of transformer blocks to stack

            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.

            transformer_context_length (int): expected length of input sequences

            transformer_embedding_dropout (float): dropout probability for embedding inputs in transformer

            transformer_block_attention_dropout (float): dropout probability for attention outputs for each transformer block

            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block

            transformer_condition_on_actions (bool): if True, transformer will condition on the sequence of actions
                in addition to the observation sequence. In this case, the inputs to the policy are expected to
                contain an "action" key in the observation dictionaries.

            transformer_predict_obs (bool): not supported (yet)

            transformer_relative_timestep (bool): if True, timesteps range from 0 to context length - 1, if False use absolute position in trajectory.

            transformer_max_timestep (int): when using absolute timesteps or euclidean distance timesteps,
                define the maximal timestep value

            transformer_euclidean_distance_timestep (int): if True, use cumulative end-effector distance traveled as timesteps

            transformer_sinusoidal_embedding (bool): if True, use sinusoidal positional embeddings that are not learned

            transformer_use_custom_transformer_block (bool): if True, use custom transformer block

            transformer_activation (str): string denoting the activation function to use in each transformer block

            transformer_nn_parameter_for_timesteps (bool): if True, use nn.Parameter for embedding timesteps

            transformer_task_id_embed_dim (int): use nn.Embedding to embed task ids

            transformer_num_task_ids (int): number of tasks we are training with

            transformer_language_enabled (bool): if True, condition on language embeddings

            transformer_language_embedding (str): string denoting the language embedding to use

            transformer_finetune_language_embedding (bool): if True, finetune the language embedding

            num_classes (int): number of classes per action dimension

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to Categorical actor
        self.num_classes = num_classes

        self.transformer_use_custom_transformer_block = transformer_use_custom_transformer_block

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        super(TransformerCategoricalActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            transformer_embed_dim=transformer_embed_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_context_length=transformer_context_length,
            transformer_embedding_dropout=transformer_embedding_dropout,
            transformer_block_attention_dropout=transformer_block_attention_dropout,
            transformer_block_output_dropout=transformer_block_output_dropout,
            transformer_condition_on_actions=transformer_condition_on_actions,
            transformer_predict_obs=transformer_predict_obs,
            transformer_relative_timestep=transformer_relative_timestep,
            transformer_max_timestep=transformer_max_timestep,
            transformer_euclidean_distance_timestep=transformer_euclidean_distance_timestep,
            transformer_sinusoidal_embedding=transformer_sinusoidal_embedding,
            transformer_use_custom_transformer_block=transformer_use_custom_transformer_block,
            transformer_activation=transformer_activation,
            transformer_nn_parameter_for_timesteps=transformer_nn_parameter_for_timesteps,
            transformer_task_id_embed_dim=transformer_task_id_embed_dim,
            transformer_num_task_ids=transformer_num_task_ids,
            transformer_language_enabled=transformer_language_enabled,
            transformer_language_embedding=transformer_language_embedding,
            transformer_finetune_language_embedding=transformer_finetune_language_embedding,
            encoder_kwargs=encoder_kwargs,
            goal_shapes=goal_shapes,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_Transformer superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        output_shapes = OrderedDict(
            logits=(self.ac_dim, self.num_classes),
        )
        if self.transformer_predict_obs:
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    output_shapes[k + "_logits"] = (v[0], self.num_classes)
        return output_shapes

    def build_dist(self, logits):
        dists = torch.distributions.Independent(
            torch.distributions.categorical.Categorical(
                logits=logits,
            ),
            1,
        )
        return dists

    def forward_train(self, obs_dict, actions=None, goal_dict=None):
        """
        Return full Categorical distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions or self.transformer_predit_obs is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        if self.transformer_condition_on_actions:
            forward_kwargs["obs"]["action"] = actions

        outputs = MIMO_Transformer.forward(self, forward_kwargs)

        dists = self.build_dist(outputs["logits"])

        if self.transformer_predict_obs:
            dists = {"action": dists}
            for k, v in self.obs_shapes.items():
                if k != "timesteps":
                    dists[k] = self.build_dist(
                        outputs[k + "_logits"],
                    )
        return dists

    def forward(self, obs_dict, actions=None, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            actions (torch.Tensor): batch of actions - only required
                if @self.transformer_condition_on_actions is True
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(obs_dict=obs_dict, actions=actions, goal_dict=goal_dict)
        if self.transformer_predict_obs:
            return {"action": out["action"].mode}
        else:
            return out.mode

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}, num_classes={}".format(
            self.ac_dim,
            self.num_classes,
        )
        return msg
