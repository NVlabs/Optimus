# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
Implementation of Behavioral Cloning (BC). Adapted from Ajay Mandlekar's private version of robomimic.
"""
import copy
from collections import OrderedDict

import numpy as np
import robomimic.models.base_nets as BaseNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import torch.distributions as D
import torch.nn as nn
from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.algo.bc import BC, BC_GMM, BC_RNN, BC_RNN_GMM, BC_VAE, BC_Gaussian

from optimus.models.transformer import TransformerActorNetwork, TransformerGMMActorNetwork


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = "gaussian" in algo_config and algo_config.gaussian.enabled
    gmm_enabled = "gmm" in algo_config and algo_config.gmm.enabled
    vae_enabled = "vae" in algo_config and algo_config.vae.enabled
    rnn_enabled = "rnn" in algo_config and algo_config.rnn.enabled
    transformer_enabled = "transformer" in algo_config and algo_config.transformer.enabled

    # enforce options don't conflict
    assert sum([gaussian_enabled, gmm_enabled, vae_enabled]) <= 1
    assert sum([rnn_enabled, transformer_enabled]) <= 1

    if rnn_enabled:
        if gmm_enabled:
            return BC_RNN_GMM, {}
        return BC_RNN, {}
    if transformer_enabled:
        if gmm_enabled:
            return BC_Transformer_GMM, {}
        return BC_Transformer, {}
    if gaussian_enabled:
        return BC_Gaussian, {}
    if gmm_enabled:
        return BC_GMM, {}
    if vae_enabled:
        return BC_VAE, {}
    return BC, {}


def transformer_args_from_config(transformer_config):
    """
    Takes a Config object corresponding to transformer settings
    (for example `config.algo.transformer` in BCConfig) and extracts
    transformer kwargs for instantiating transformer networks.
    """
    return dict(
        transformer_num_layers=transformer_config.num_layers,
        transformer_context_length=transformer_config.context_length,
        transformer_embed_dim=transformer_config.embed_dim,
        transformer_num_heads=transformer_config.num_heads,
        transformer_embedding_dropout=transformer_config.embedding_dropout,
        transformer_block_attention_dropout=transformer_config.block_attention_dropout,
        transformer_block_output_dropout=transformer_config.block_output_dropout,
        layer_dims=transformer_config.layer_dims,
    )


class BC(PolicyAlgo):
    """
    Normal BC training.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        return TensorUtils.to_float(input_batch)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])
        losses["l2_loss_eef"] = nn.MSELoss()(actions[..., :3], a_target[..., :3])
        losses["l2_loss_ori"] = nn.MSELoss()(actions[..., 3:6], a_target[..., 3:6])
        losses["l2_loss_is_gripper"] = nn.MSELoss()(actions[..., 6], a_target[..., 6])
        losses["l2_loss_grasp"] = nn.MSELoss()(actions[..., 7], a_target[..., 7])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss_eef" in info["losses"]:
            log["L2_Loss_EEF"] = info["losses"]["l2_loss_eef"].item()
        if "l2_loss_ori" in info["losses"]:
            log["L2_Loss_Ori"] = info["losses"]["l2_loss_ori"].item()
        if "l2_loss_is_gripper" in info["losses"]:
            log["L2_Loss_Is_Gripper"] = info["losses"]["l2_loss_is_gripper"].item()
        if "l2_loss_grasp" in info["losses"]:
            log["L2_Loss_Grasp"] = info["losses"]["l2_loss_grasp"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
            actions=dists.mean,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        a_target = batch["actions"]
        actions = predictions["actions"]
        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        losses = OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

        losses["l2_loss_eef"] = nn.MSELoss()(actions[..., :3], a_target[..., :3])
        losses["l2_loss_ori"] = nn.MSELoss()(actions[..., 3:6], a_target[..., 3:6])
        losses["l2_loss_is_gripper"] = nn.MSELoss()(actions[..., 6], a_target[..., 6])
        # losses["l2_loss_grasp"] = nn.MSELoss()(actions[..., 7], a_target[..., 7]) # this is hardcoded for 7dof ee-ctrl actions
        return losses

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "l2_loss_eef" in info["losses"]:
            log["L2_Loss_EEF"] = info["losses"]["l2_loss_eef"].item()
        if "l2_loss_ori" in info["losses"]:
            log["L2_Loss_Ori"] = info["losses"]["l2_loss_ori"].item()
        if "l2_loss_is_gripper" in info["losses"]:
            log["L2_Loss_Is_Gripper"] = info["losses"]["l2_loss_is_gripper"].item()
        if "l2_loss_grasp" in info["losses"]:
            log["L2_Loss_Grasp"] = info["losses"]["l2_loss_grasp"].item()
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.epoch_every_n_steps = self.global_config.experiment.epoch_every_n_steps
        self.num_epochs = self.global_config.train.num_epochs


class BC_RNN(BC_RNN):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)
        self.lr_warmup = False

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        return input_batch

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(
                batch_size=batch_size, device=list(obs_dict.values())[0].device
            )

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state
        )
        return action


class BC_RNN_GMM(BC_RNN_GMM):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)
        self.lr_warmup = False
        self.epoch_every_n_steps = self.global_config.experiment.epoch_every_n_steps
        self.num_epochs = self.global_config.train.num_epochs

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        return input_batch

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(
                batch_size=batch_size, device=list(obs_dict.values())[0].device
            )

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state
        )
        return action


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """

    def set_params_from_config(self):
        # read specific config variables we need for training / eval
        self.context_length = self.algo_config.transformer.context_length
        self.optimizer_type = self.algo_config.transformer.optimizer_type
        self.lr_scheduler_type = self.algo_config.transformer.lr_scheduler_type
        self.lr_warmup = self.algo_config.transformer.lr_warmup
        self.num_open_loop_actions_to_execute = (
            self.algo_config.transformer.num_open_loop_actions_to_execute
        )
        self.epoch_every_n_steps = self.global_config.experiment.epoch_every_n_steps
        self.num_epochs = self.global_config.train.num_epochs

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **transformer_args_from_config(self.algo_config.transformer),
        )

        self.set_params_from_config()
        self.algo_config = None
        self.obs_config = None
        self.global_config = None

    def linear_schedule_by_factor_10_over_100(self, step):
        # scales down learning rate by 10^-1 after 100 epochs
        epoch = step // self.epoch_every_n_steps
        if epoch < 100:
            return (-0.009) * float(epoch) + 1
        else:
            return (-0.009) * float(100) + 1

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        if self.optimizer_type == "adamw":
            self.optimizers["policy"] = torch.optim.AdamW(
                self.nets.parameters(),
                lr=self.optim_params["policy"]["learning_rate"]["initial"],
                betas=self.optim_params["policy"]["learning_rate"]["betas"],
                weight_decay=self.optim_params["policy"]["learning_rate"]["decay_factor"],
            )
        elif self.optimizer_type == "adam":
            self.optimizers["policy"] = torch.optim.AdamW(
                self.nets.parameters(),
                lr=self.optim_params["policy"]["learning_rate"]["initial"],
                betas=self.optim_params["policy"]["learning_rate"]["betas"],
            )
        if self.lr_scheduler_type == "cosine":
            self.lr_schedulers["policy"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers["policy"],
                T_max=int(1e6),
                eta_min=self.optim_params["policy"]["learning_rate"]["initial"] * 1 / 10,
            )
        elif self.lr_scheduler_type == "linear":
            self.lr_schedulers["policy"] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers["policy"], self.linear_schedule_by_factor_10_over_100
            )
        elif self.lr_scheduler_type == "none":
            self.lr_schedulers["policy"] = None
        elif self.lr_scheduler_type == "linear_to_0":
            self.lr_schedulers["policy"] = torch.optim.lr_scheduler.LinearLR(
                self.optimizers["policy"],
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.num_epochs * self.epoch_every_n_steps,
            )

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # because of PL: cast to device here:
            batch = TensorUtils.to_device(TensorUtils.to_float(batch), self.device)
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses, epoch)
                info.update(step_info)

        return info

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        # just use final timestep
        input_batch["actions"] = batch["actions"][:, -1, :]
        return input_batch

    def _forward_training(self, batch):
        """
        Modify from super class to handle different inputs and outputs (e.g. conditioning on actions).
        """

        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
                self.context_length
            ),
        )

        predictions = OrderedDict()

        masked_obs, masked_actions = batch["obs"], batch["actions"]

        predictions["actions"] = self.nets["policy"](
            obs_dict=masked_obs, actions=masked_actions, goal_dict=batch["goal_obs"]
        )

        # only supervise final timestep
        predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        TensorUtils.assert_size_at_dim(
            obs_dict,
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
                self.context_length
            ),
        )
        actions = obs_dict["actions"]
        del obs_dict["actions"]

        out = self.nets["policy"](obs_dict, actions=actions, goal_dict=goal_dict)
        action = out[:, -1, :]
        return action

    def _train_step(self, losses, epoch):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        # gradient step
        info = OrderedDict()

        optim = self.optimizers["policy"]
        loss = losses["action_loss"]
        net = self.nets["policy"]
        max_grad_norm = None

        # backprop
        optim.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

        # compute grad norms
        grad_norms = 0.0
        for p in net.parameters():
            # only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                grad_norms += p.grad.data.norm(2).pow(2).item()

        # step
        for pg in optim.param_groups:
            if epoch <= 100:
                pg["lr"] = (-0.009 * float(epoch) + 1) * self.optim_params["policy"][
                    "learning_rate"
                ]["initial"]
            else:
                pg["lr"] = 1e-5
        optim.step()
        info["policy_grad_norms"] = grad_norms
        return info

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            # self.algo_config.loss.l2_weight * losses["l2_loss"],
            losses["l2_loss"],
            # self.algo_config.loss.l1_weight * losses["l1_loss"],
            # self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses


class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **transformer_args_from_config(self.algo_config.transformer),
        )
        self.set_params_from_config()
        self.algo_config = None
        self.obs_config = None
        self.global_config = None

    def _forward_training(self, batch):
        """
        Modify from super class to support GMM training.
        """

        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
                self.context_length
            ),
        )

        masked_obs, masked_actions = batch["obs"], batch["actions"]

        dists = self.nets["policy"].forward_train(
            obs_dict=masked_obs,
            actions=masked_actions,
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]

        # only use final timestep prediction by making a new distribution with only final timestep.
        # This essentially does `dists = dists[:, -1]`
        component_distribution = D.Normal(
            loc=dists.component_distribution.base_dist.loc[:, -1],
            scale=dists.component_distribution.base_dist.scale[:, -1],
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
