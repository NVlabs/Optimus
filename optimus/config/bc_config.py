# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
Config for BC algorithm. Taken from Ajay Mandlekar's private version of robomimic.
"""

from robomimic.config.base_config import BaseConfig


class BCConfig(BaseConfig):
    ALGO_NAME = "bc"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength
        self.algo.optim_params.policy.learning_rate.betas = (0.9, 0.99)
        self.algo.optim_params.policy.learning_rate.lr_scheduler_interval = "epoch"

        # loss weights
        self.algo.loss.l2_weight = 1.0  # L2 loss weight
        self.algo.loss.l1_weight = 0.0  # L1 loss weight
        self.algo.loss.cos_weight = 0.0  # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False  # whether to train a Gaussian policy
        self.algo.gaussian.fixed_std = False  # whether to train std output or keep it constant
        self.algo.gaussian.init_std = 0.1  # initial standard deviation (or constant)
        self.algo.gaussian.min_std = 0.01  # minimum std output from network
        self.algo.gaussian.std_activation = (
            "softplus"  # activation to use for std output from policy net
        )
        self.algo.gaussian.low_noise_eval = True  # low-std at test-time

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False  # whether to train a GMM policy
        self.algo.gmm.num_modes = 5  # number of GMM modes
        self.algo.gmm.min_std = 0.0001  # minimum std output from network
        self.algo.gmm.std_activation = (
            "softplus"  # activation to use for std output from policy net
        )
        self.algo.gmm.low_noise_eval = True  # low-std at test-time

        # stochastic VAE policy settings
        self.algo.vae.enabled = False  # whether to train a VAE policy
        self.algo.vae.latent_dim = (
            14  # VAE latent dimnsion - set to twice the dimensionality of action space
        )
        self.algo.vae.latent_clip = None  # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = (
            1.0  # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO
        )

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = (
            True  # whether decoder should condition on observation
        )
        self.algo.vae.decoder.reconstruction_sum_across_elements = (
            False  # sum instead of mean for reconstruction loss
        )

        # VAE prior settings
        self.algo.vae.prior.learn = False  # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False  # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False  # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10  # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False  # whether to learn GMM weights
        self.algo.vae.prior.use_categorical = False  # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = (
            10  # the number of categorical classes for each latent dimension
        )
        self.algo.vae.prior.categorical_gumbel_softmax_hard = (
            False  # use hard selection in forward pass
        )
        self.algo.vae.prior.categorical_init_temp = 1.0  # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001  # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3  # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)  # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)  # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (
            300,
            400,
        )  # prior MLP layer dimensions (if learning conditioned prior)

        # RNN policy settings
        self.algo.rnn.enabled = False  # whether to train RNN policy
        self.algo.rnn.horizon = 10  # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400  # hidden dimension size
        self.algo.rnn.rnn_type = "LSTM"  # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2  # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False  # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False  # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Transformer policy settings
        self.algo.transformer.enabled = False  # whether to train transformer policy
        self.algo.transformer.context_length = 64  # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.transformer.embed_dim = 256  # dimension for embeddings used by transformer
        self.algo.transformer.num_layers = 6  # number of transformer blocks to stack
        self.algo.transformer.num_heads = 8  # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.transformer.embedding_dropout = (
            0.1  # dropout probability for embedding inputs in transformer
        )
        self.algo.transformer.block_attention_dropout = (
            0.1  # dropout probability for attention outputs for each transformer block
        )
        self.algo.transformer.block_output_dropout = (
            0.1  # dropout probability for final outputs for each transformer block
        )
        self.algo.transformer.condition_on_actions = False  # whether to condition on the sequence of past actions in addition to the observation sequence
        self.algo.transformer.predict_obs = (
            False  # whether to predict observations in the output sequences as well
        )
        self.algo.transformer.mask_inputs = False  # whether to use bert style input masking
        self.algo.transformer.relative_timestep = True  # if true timesteps range from 0 to context length-1, if false use absolute position in trajectory
        self.algo.transformer.euclidean_distance_timestep = False  # if true timesteps are based the cumulative distance traveled by the end effector. otherwise integer timesteps
        self.algo.transformer.max_timestep = (
            1250  # for the nn.embedding layer, must know the maximal timestep value
        )
        self.algo.transformer.open_loop_predictions = (
            False  # if true don't run transformer at every step, execute a set of predicted actions
        )
        self.algo.transformer.sinusoidal_embedding = (
            False  # if True, use standard positional encodings (sin/cos)
        )
        self.algo.transformer.obs_noise_scale = (
            0.05  # amount of noise to add to the observations during training
        )
        self.algo.transformer.add_noise_to_train_obs = (
            False  # if true add noise to the observations during training
        )
        self.algo.transformer.use_custom_transformer_block = (
            True  # if True, use custom transformer block
        )
        self.algo.transformer.optimizer_type = "adamw"  # toggle the type of optimizer to use
        self.algo.transformer.lr_scheduler_type = "linear"  # toggle the type of lr_scheduler to use
        self.algo.transformer.lr_warmup = (
            False  # if True, warmup the learning rate from some small value
        )
        self.algo.transformer.activation = (
            "gelu"  # activation function for MLP in Transformer Block
        )
        self.algo.transformer.num_open_loop_actions_to_execute = (
            10  # number of actions to execute in open loop
        )
        self.algo.transformer.supervise_all_steps = (
            False  # if true, supervise all intermediate actions, otherwise only final one
        )
        self.algo.transformer.nn_parameter_for_timesteps = (
            True  # if true, use nn.Parameter otherwise use nn.Embedding
        )
        self.algo.transformer.num_task_ids = 1  # number of tasks we are training with
        self.algo.transformer.task_id_embed_dim = 0  # dimension of the task id embedding
        self.algo.transformer.language_enabled = False  # if true, condition on language embeddings
        self.algo.transformer.language_embedding = (
            "raw"  # string denoting the language embedding to use
        )
        self.algo.transformer.finetune_language_embedding = (
            False  # if true, finetune the language embedding
        )
        self.algo.transformer.kl_loss_weight = 0  # 5e-6  # weight of the KL loss
        self.algo.transformer.use_cvae = True
        self.algo.transformer.predict_signature = False
        self.algo.transformer.layer_dims = (1024, 1024)
        self.algo.transformer.latent_dim = 0
        self.algo.transformer.prior_use_gmm = True
        self.algo.transformer.prior_gmm_num_modes = 10
        self.algo.transformer.prior_gmm_learn_weights = True
        self.algo.transformer.replan_every_step = False
        self.algo.transformer.decoder = False
        self.algo.transformer.prior_use_categorical = False
        self.algo.transformer.prior_categorical_gumbel_softmax_hard = False
        self.algo.transformer.prior_categorical_dim = 10
        self.algo.transformer.primitive_type = "none"
        self.algo.transformer.reset_context_after_primitive_exec = True
        self.algo.transformer.block_drop_path = 0.0
        self.algo.transformer.use_cross_attention_conditioning = False
        self.algo.transformer.use_alternating_cross_attention_conditioning = False
        self.algo.transformer.key_value_from_condition = False
        self.algo.transformer.add_primitive_id = False
        self.algo.transformer.tokenize_primitive_id = False
        self.algo.transformer.channel_condition = False
        self.algo.transformer.tokenize_obs_components = False
        self.algo.transformer.num_patches_per_image_dim = 1
        self.algo.transformer.nets_to_freeze = ()
        self.algo.transformer.use_ndp_decoder = False
        self.algo.transformer.ndp_decoder_kwargs = None
        self.algo.transformer.transformer_type = "gpt"
        self.algo.transformer.mega_kwargs = {}

        self.algo.corner_loss.enabled = False

        self.algo.dml.enabled = (
            False  # if True, use Discretized Mixture of Logistics output distribution
        )
        self.algo.dml.num_modes = 2  # number of modes in the DML distribution
        self.algo.dml.num_classes = (
            256  # number of classes in each dimension of the discretized action space
        )
        self.algo.dml.log_scale_min = -7.0  # minimum value of the log scale
        self.algo.dml.constant_variance = True  # if True, use a constant variance

        self.algo.cat.enabled = False  # if True, use categorical output distribution
        self.algo.cat.num_classes = (
            256  # number of classes in each dimension of the categorical action space
        )

        self.env_seed = 0  # seed for environment
        self.wandb_project_name = "test"
        self.train.fast_dev_run = False  # whether to run training in debug mode
        self.train.max_grad_norm = None  # max gradient norm
        self.train.amp_enabled = False  # use automatic mixed precision
        self.train.num_gpus = 1  # number of gpus to use
        self.experiment.rollout.goal_conditioning_enabled = (
            False  # if True, use goal conditioning wrapper to sample goals for inference
        )
        self.train.load_next_obs = False  # whether or not to load s'
        self.experiment.rollout.goal_success_threshold = 0.01  # if the distance from the goal is less than goal_success_threshold, this counts as a success
        self.train.pad_frame_stack = True
        self.train.pad_seq_length = True
        self.train.frame_stack = 1
        self.experiment.rollout.is_mujoco = False
        self.experiment.rollout.valid_key = "valid"
        self.train.ckpt_path = None
        self.train.use_swa = False
        self.experiment.rollout.parallel_rollouts = True
        self.experiment.rollout.select_random_subset = False
        self.train.save_ckpt_on_epoch_end = True

        self.algo.dagger.enabled = False  # if true, enable dagger support
        self.algo.dagger.online_epoch_rate = 50  # how often to collect online data
        self.algo.dagger.num_rollouts = 1  # number of rollouts to collect per online epoch
        self.algo.dagger.rollout_type = (
            "state_error_mixed"  # toggle rollout type, can range from policy to closed loop tamp
        )
        self.algo.dagger.state_error_threshold = (
            0.001  # threshold for state error - triggers TAMP solver action
        )
        self.algo.dagger.action_error_threshold = (
            0.001  # threshold for action error - triggers TAMP solver action
        )
        self.algo.dagger.mpc_horizon = (
            100  # MPC horizon for closed loop tamp MPC - useful for running faster
        )
