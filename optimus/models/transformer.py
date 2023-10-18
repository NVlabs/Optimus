# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Main Transformer implementation in OPTIMUS.
"""

import math
import textwrap
from collections import OrderedDict

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
from robomimic.models.base_nets import Module
from robomimic.models.distributions import TanhWrappedDistribution
from torch.nn import functional as F

from optimus.models.obs_nets import (
    ObservationDecoder,
    ObservationGroupEncoder,
)


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

    def forward(self, x):
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
        """
        super(Transformer_Block, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.nets = nn.ModuleDict()

        self.nets["attention"] = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            attention_dropout=attention_dropout,
            output_dropout=output_dropout,
        )

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

    def forward(self, inputs):
        """
        Forward pass - chain self-attention + MLP blocks, with residual connections and layer norms.
        """
        x = inputs["x"]
        x = x + self.nets["attention"](self.nets["ln1"](x))
        x = x + self.nets["mlp"](self.nets["ln2"](x))
        return {"x": x}

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

            activation (str): string denoting the activation function to use in each transformer block

        """
        super(GPT_Backbone, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.block_attention_dropout = block_attention_dropout
        self.block_output_dropout = block_output_dropout
        self.block_drop_path = block_drop_path

        self.activation = nn.GELU()

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
        self.nets["transformer"] = nn.Sequential(
            *[
                Transformer_Block(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    context_length=self.context_length,
                    attention_dropout=self.block_attention_dropout,
                    output_dropout=self.block_output_dropout,
                    activation=self.activation,
                )
                for i in range(self.num_layers)
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

    def forward(self, inputs):
        assert inputs.shape[1:] == (self.context_length, self.embed_dim), inputs.shape
        x = self.nets["transformer"]({"x": inputs})["x"]
        transformer_output = self.nets["output_ln"](x)
        return transformer_output


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
        layer_dims=(1024, 1024),
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

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

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
        self.obs_encoder_output_dim = transformer_embed_dim
        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.layer_dims = layer_dims

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

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
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=self.transformer_context_length,
            block_attention_dropout=transformer_block_attention_dropout,
            block_output_dropout=transformer_block_output_dropout,
            block_drop_path=transformer_block_drop_path,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
        )

    def build_positional_embedding_nets(self):
        max_timestep = self.transformer_context_length

        self.params["embed_timestep"] = nn.Parameter(
            torch.zeros(1, max_timestep, self.transformer_embed_dim)
        )

    def input_embedding(self, inputs):
        if type(inputs) is not dict:
            inputs = {"obs": {"all": inputs}}
        embeddings = {
            k: self.nets["embed_encoder"][k](inputs["obs"][k]) for k in self.obs_components_keys
        }

        new_embeddings = OrderedDict()
        for idx, k in enumerate(self.obs_components_keys):
            v = embeddings[k]
            time_embeddings = self.params["embed_timestep"]
            v = v + time_embeddings
            v = self.nets["embed_ln"](v)
            v = self.nets["embed_drop"](v)
            new_embeddings[k] = v

        # embeddings are K k:v pairs with v having shapes BxTxD
        # need to cat them so that we have a single tensor of shape Bx(T*K)xD
        # this is valid because we have provided timestep embeddings to each of the K tensors
        embeddings = torch.cat([v for k, v in new_embeddings.items()], dim=1)
        return embeddings

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

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

        # do not encode timesteps through encoder network!
        del inputs["obs"]["timesteps"]

        # use encoder to each timestep of sequence to extract flat transformer inputs
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        # assert transformer_inputs.ndim == 3  # [B, T, D]

        transformer_embeddings = self.input_embedding(transformer_inputs)
        # pass encoded sequences through transformer
        transformer_outputs = self.nets["transformer"].forward(transformer_embeddings)

        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        return transformer_outputs

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
        layer_dims=(1024, 1024),
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

            use_cvae (bool): if True, use condition on initial obs for the prior and encoder

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

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

        # set up different observation groups for @Transformer_MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        del observation_group_shapes["obs"]["timesteps"]

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
            layer_dims=layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_Transformer, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        output_shapes = OrderedDict(action=(self.ac_dim,))
        return output_shapes

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        output_shapes = {k: list(self.output_shapes[k]) for k in self.output_shapes}
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
            outputs (torch.Tensor): contains predicted action sequence
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        forward_kwargs = dict(obs=obs_dict, goal=goal_dict)
        outputs = super(TransformerActorNetwork, self).forward(forward_kwargs)

        # apply tanh squashing to ensure actions are in [-1, 1]
        outputs["action"] = torch.tanh(outputs["action"])

        return outputs["action"]  # only action sequences

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


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
        layer_dims=(1024, 1024),
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

            layer_dims ([int]): sequence of integers for the encoder hidden
                layer sizes.

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
            layer_dims=layer_dims,
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

        outputs = MIMO_Transformer.forward(self, forward_kwargs)

        dists = self.build_dist(
            outputs["mean"], outputs["scale"], outputs["logits"], use_tanh=self.use_tanh
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
