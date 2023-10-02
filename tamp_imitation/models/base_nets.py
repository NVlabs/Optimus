# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import textwrap

import numpy as np
import torch
import torch.nn as nn
from robomimic.models.base_nets import *
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict


class ViTBase(ConvBase):
    def __init__(
        self,
        image_size=76,
        patch_size=19,
        feature_dimension=64,
        dim=256,
        depth=4,
        heads=4,
        mlp_dim=256 * 4,
        dropout=0.1,
        emb_dropout=0.1,
        patchify_scheme="flatten",
    ):
        super(ViTBase, self).__init__()
        self.net = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=feature_dimension,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            patchify_scheme=patchify_scheme,
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        return [64]

    def forward(self, inputs):
        return self.net(inputs)

    def __repr__(self):
        """Pretty print network."""
        return str(self.net)


class VisualCore(EncoderCore, ConvBase):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """

    def __init__(
        self,
        input_shape,
        backbone_class,
        backbone_kwargs,
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network (e.g.: ResNet18)
            backbone_kwargs (dict): kwargs for the visual backbone network
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual feature
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisualCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(
            cls=eval(backbone_class), dic=backbone_kwargs, copy=True
        )

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, ConvBase)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # maybe make pool net
        if pool_class is not None:
            assert isinstance(pool_class, str)
            # feed output shape of backbone to pool net
            if pool_kwargs is None:
                pool_kwargs = dict()
            # extract only relevant kwargs for this specific backbone
            pool_kwargs["input_shape"] = feat_shape
            pool_kwargs = extract_class_init_kwargs_from_dict(
                cls=eval(pool_class), dic=pool_kwargs, copy=True
            )
            self.pool = eval(pool_class)(**pool_kwargs)
            assert isinstance(self.pool, Module)

            feat_shape = self.pool.output_shape(feat_shape)
            net_list.append(self.pool)
        else:
            self.pool = None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(
                self.input_shape, self.output_shape(self.input_shape)
            ),
            indent,
        )
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + "(" + msg + "\n)"
        return msg
