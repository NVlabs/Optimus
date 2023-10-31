# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
from robomimic.models.base_nets import *
from robomimic.models.obs_nets import *


def obs_encoder_factory(
    obs_shapes,
    feature_activation=nn.ReLU,
    encoder_kwargs=None,
    return_dict_features=False,
    num_patches_per_image_dim=1,
):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
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
        num_patches_per_image_dim (int): number of patches to extract from each image dimension. If 1, no patches.
        return_dict_features: instead of concatenating all features together, return a dictionary
    """
    enc = ObservationEncoder(
        feature_activation=feature_activation,
        return_dict_features=return_dict_features,
        num_patches_per_image_dim=num_patches_per_image_dim,
    )
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = (
            deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality])
            if encoder_kwargs is None
            else deepcopy(encoder_kwargs[obs_modality])
        )

        for obs_module, cls_mapping in zip(
            ("core", "obs_randomizer"),
            (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS),
        ):
            # Sanity check for kwargs in case they don't exist / are None
            if enc_kwargs.get(f"{obs_module}_kwargs", None) is None:
                enc_kwargs[f"{obs_module}_kwargs"] = {}
            # Add in input shape info
            enc_kwargs[f"{obs_module}_kwargs"]["input_shape"] = obs_shape
            # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
            if enc_kwargs[f"{obs_module}_class"] is not None:
                enc_kwargs[f"{obs_module}_kwargs"] = extract_class_init_kwargs_from_dict(
                    cls=cls_mapping[enc_kwargs[f"{obs_module}_class"]],
                    dic=enc_kwargs[f"{obs_module}_kwargs"],
                    copy=False,
                )

        # Add in input shape info
        randomizer = (
            None
            if enc_kwargs["obs_randomizer_class"] is None
            else ObsUtils.OBS_RANDOMIZERS[enc_kwargs["obs_randomizer_class"]](
                **enc_kwargs["obs_randomizer_kwargs"]
            )
        )

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizer=randomizer,
        )

    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks.
    """

    def __init__(
        self,
        feature_activation=nn.ReLU,
        return_dict_features=False,
        num_patches_per_image_dim=1,
    ):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.
            num_patches_per_image_dim (int): number of patches to extract from each image dimension. If 1, no patches.
            return_dict_features: instead of concatenating all features together, return a dictionary
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self.return_dict_features = return_dict_features
        self.num_patches_per_image_dim = num_patches_per_image_dim
        self._locked = False

    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizer=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(
            name
        )

        if net is not None:
            assert isinstance(
                net, Module
            ), "ObservationEncoder: @net must be instance of Module class"
            assert (
                (net_class is None) and (net_kwargs is None) and (share_net_from is None)
            ), "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        if randomizer is not None:
            assert isinstance(randomizer, Randomizer)
            if net_kwargs is not None:
                # update input shape to visual core
                net_kwargs["input_shape"] = randomizer.output_shape_in(shape)

        if name.endswith("image"):
            # update input shape to match patch size
            input_shape = net_kwargs["input_shape"]
            net_kwargs["input_shape"] = (
                input_shape[0],
                input_shape[1] // self.num_patches_per_image_dim,
                input_shape[2] // self.num_patches_per_image_dim,
            )

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = randomizer
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](
                    **self.obs_nets_kwargs[k]
                )
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(
            obs_dict
        ), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        if self.return_dict_features:
            feats = OrderedDict()
        else:
            feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            if k.endswith("image") and self.return_dict_features:
                patch_size = x.shape[-1] // self.num_patches_per_image_dim

                # split image into patches of size patch_size x patch_size
                # x shape = [B, C, H, W]
                # patched_x shape = [B, C, num_patches_per_image_dim, num_patches_per_image_dim, patch_size, patch_size]
                patched_x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

                # move the patch dimensions next to the batch dimension
                patched_x = patched_x.permute(0, 2, 3, 1, 4, 5)
                shape = patched_x.shape

                # flatten the patch dimensions into the batch dimension
                # batched_patched_x shape = [B * num_patches_per_image_dim * num_patches_per_image_dim, C, patch_size, patch_size]
                batched_patched_x = patched_x.reshape(
                    shape[0] * shape[1] * shape[2], shape[3], shape[4], shape[5]
                )
                x = batched_patched_x
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k](x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            x = TensorUtils.flatten(x, begin_axis=1)
            if self.return_dict_features:
                if k.endswith("image"):
                    assert (x[0] - x[1]).norm() > 0
                    # unflatten the batch dimension into batfch dimension and total number of patches
                    # x shape = [B, num_patches_per_image_dim * num_patches_per_image_dim, D]
                    x = x.reshape(
                        shape[0],
                        self.num_patches_per_image_dim * self.num_patches_per_image_dim,
                        -1,
                    )
                    for i in range(self.num_patches_per_image_dim * self.num_patches_per_image_dim):
                        feats[k + "_patch_" + str(i)] = x[:, i, :]
                else:
                    feats[k] = x
            else:
                feats.append(x)

        if self.return_dict_features:
            return feats
        else:
            # concatenate all features together
            return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        if self.return_dict_features:
            feat_dict = {}
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
            if self.return_dict_features:
                feat_dict[k] = int(np.prod(feat_shape))
        if self.return_dict_features:
            return feat_dict
        else:
            return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent("randomizer={}\n".format(self.obs_randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", " " * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """

    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
        return_dict_features=False,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

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
            return_dict_features: instead of concatenating all features together, return a dictionary
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all(
            [isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes]
        )

        self.observation_group_shapes = observation_group_shapes
        self.return_dict_features = return_dict_features

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
                return_dict_features=return_dict_features,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(
            inputs
        ), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        if self.return_dict_features:
            outputs = {}
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            output = self.nets[obs_group].forward(inputs[obs_group])
            if self.return_dict_features:
                outputs[obs_group] = output
            else:
                outputs.append(output)
        if self.return_dict_features:
            return outputs
        else:
            return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        if self.return_dict_features:
            feat_dict = {}
        for obs_group in self.observation_group_shapes:
            if self.return_dict_features:
                feat_dict[obs_group] = self.nets[obs_group].output_shape()
            else:
                # get feature dimension of these keys
                feat_dim += self.nets[obs_group].output_shape()[0]
        if self.return_dict_features:
            return feat_dict
        else:
            return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.observation_group_shapes:
            msg += "\n"
            indent = " " * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class SplitObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. This subclass is used
    to handle outputting a dictionary.
    """

    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim // len(self.obs_shapes), layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.obs_shapes[k]) for k in self.obs_shapes}

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        split_feats = torch.chunk(feats, len(self.obs_shapes))
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](split_feats[k])
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg
