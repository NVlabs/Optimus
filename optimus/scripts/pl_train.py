# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
The main entry point for training policies. Adapted to use PyTorch Lightning and Optimus codebase.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes
"""
import argparse
import json
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np
import psutil
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
import wandb
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, StochasticWeightAveraging, Timer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.strategies import DDPStrategy
from robomimic.algo import RolloutPolicy
from robomimic.algo.algo import PolicyAlgo
from robomimic.utils.log_utils import PrintLogger
from torch.utils.data import DataLoader

import optimus.utils.env_utils as EnvUtils
import optimus.utils.file_utils as FileUtils
import optimus.utils.train_utils as TrainUtils
from optimus.algo import algo_factory
from optimus.config import config_factory
from optimus.envs.wrappers import EvaluateOnDatasetWrapper, FrameStackWrapper


class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader = DataLoader(dataset=self.train_dataset, **self.train_dataloader_params)
        return new_dataloader


class RolloutCallback(Callback):
    """
    Callback that handles running rollouts and saving checkpoints based on best return.
    Also handles online epoch rollouts when toggled.
    """

    def __init__(
        self,
        envs,
        video_dir,
        obs_normalization_stats,
        trainset,
        ckpt_dir,
        vis_data,
        config_path,
        rollout_dir,
        exp_log_dir,
    ):
        config = TrainUtils.get_config_from_path(config_path)
        self.envs = envs
        self.video_dir = video_dir if config.experiment.render_video else None
        self.rollout_dir = rollout_dir
        self.exp_log_dir = exp_log_dir
        self.obs_normalization_stats = obs_normalization_stats
        self.best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
        self.best_success_rate = (
            {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
        )
        self.trainset = trainset
        self.ckpt_dir = ckpt_dir
        self.data = (
            {k: dict() for k in envs} if config.experiment.rollout.enabled else None
        )  # store all the scalar data logged so far
        self.vis_data = vis_data
        self.config_path = config_path

    def load_state_dict(self, state_dict):
        self.best_return = state_dict["best_return"]
        self.best_success_rate = state_dict["best_success_rate"]
        self.obs_normalization_stats = state_dict["obs_normalization_stats"]
        self.data = state_dict["data"]

    def state_dict(self):
        return dict(
            best_return=self.best_return,
            best_success_rate=self.best_success_rate,
            obs_normalization_stats=self.obs_normalization_stats,
            data=self.data,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        unused=0,
    ):
        config = TrainUtils.get_config_from_path(self.config_path)
        epoch = pl_module.global_step // config.experiment.epoch_every_n_steps
        envs = self.envs
        video_dir = self.video_dir
        model = pl_module.model
        rollout_check = epoch % config.experiment.rollout.rate == 0
        epoch_ckpt_name = "model_epoch_{}".format(epoch)
        epoch_end = pl_module.global_step % config.experiment.epoch_every_n_steps == 0
        exp_log_dir = self.exp_log_dir
        epoch_exp_log_dir = os.path.join(exp_log_dir, f"epoch_{epoch}")
        os.makedirs(epoch_exp_log_dir, exist_ok=True)
        if (
            config.experiment.rollout.enabled
            and (epoch > config.experiment.rollout.warmstart)
            and rollout_check
            and epoch_end
        ):
            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(
                model, obs_normalization_stats=self.obs_normalization_stats
            )
            rollout_model.policy.device = pl_module.device
            num_episodes = config.experiment.rollout.n
            (all_rollout_logs, video_paths) = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
                rollout_dir=self.rollout_dir,
                config=config,
            )

            if video_dir:
                for k in video_paths.keys():
                    trainer.logger.experiment.log(
                        {f"{k}video": wandb.Video(video_paths[k], format="mp4")}
                    )

            # summarize results from rollouts to wandb and terminal
            cumulative_success_rate = 0
            cumulative_solve_rate = 0
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k not in self.data[env_name]:
                        self.data[env_name][k] = []
                    self.data[env_name][k].append(v)
                    if k == "Success_Rate":
                        cumulative_success_rate += v
                    if k.startswith("Time_"):
                        pl_module.log(
                            "Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]),
                            v,
                        )
                    else:
                        pl_module.log("Rollout/{}/{}".format(k, env_name), v)

                    stats = self.get_stats(k, env_name)
                    for stat_k, stat_v in stats.items():
                        if k == "Success_Rate" and stat_k == "max":
                            # this means the task was solved at some checkpoint in training
                            cumulative_solve_rate += int(stat_v == 1)
                        stat_k_name = "{}-{}".format(env_name, stat_k)
                        if k.startswith("Time_"):
                            pl_module.log(
                                "Timing_Stats/Rollout_{}_{}".format(stat_k_name, k[5:]),
                                stat_v,
                            )
                        else:
                            pl_module.log("Rollout/{}/{}".format(k, stat_k_name), stat_v)
                print("Env: {}".format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))
                # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=self.best_return,
                best_success_rate=self.best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            self.best_return = updated_stats["best_return"]
            self.best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (
                config.experiment.save.enabled
                and updated_stats["should_save_ckpt"]
                or config.train.num_gpus > 1
            )

            if should_save_ckpt:
                checkpoint_path = os.path.join(self.ckpt_dir, epoch_ckpt_name + ".ckpt")
                trainer.save_checkpoint(checkpoint_path)

        model.set_train()  # rollouts disable training mode

    def on_train_epoch_end(self, trainer, pl_module):
        # every epoch, overwrite current checkpoint
        config = TrainUtils.get_config_from_path(self.config_path)
        if config.train.save_ckpt_on_epoch_end:
            checkpoint_path = os.path.join(self.ckpt_dir, "model_latest.ckpt")
            trainer.save_checkpoint(checkpoint_path)
        pass

    def get_stats(self, k, env_name):
        """
        Computes running statistics for a particular key.

        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats["mean"] = np.mean(self.data[env_name][k])
        stats["std"] = np.std(self.data[env_name][k])
        stats["min"] = np.min(self.data[env_name][k])
        stats["max"] = np.max(self.data[env_name][k])
        return stats


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    def __init__(self, model):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.model = model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except:
            pass
        self.step_log_all_train = []
        self.step_log_all_valid = []

    def training_step(self, batch, batch_idx):
        batch["obs"] = ObsUtils.process_obs_dict(batch["obs"])
        info = PolicyAlgo.train_on_batch(self.model, batch, self.current_epoch, validate=False)
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model._forward_training(batch)
        losses = self.model._compute_losses(predictions, batch)
        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_train.append(self.model.log_info(info))
        if self.global_step % self.model.epoch_every_n_steps == 0:
            # flatten and take the mean of the metrics
            log = {}
            for i in range(len(self.step_log_all_train)):
                for k in self.step_log_all_train[i]:
                    if k not in log:
                        log[k] = []
                    log[k].append(self.step_log_all_train[i][k])
            log_all = dict((k, float(np.mean(v))) for k, v in log.items())
            for k in self.model.optimizers:
                for i, param_group in enumerate(self.model.optimizers[k].param_groups):
                    log_all["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]
            for k, v in log_all.items():
                self.log("Train/" + k, v, sync_dist=True)
            self.step_log_all_train = []
        return losses["action_loss"]

    def validation_step(self, batch, batch_idx):
        try:
            low_noise_eval = self.model.nets["policy"].low_noise_eval
            self.model.nets["policy"].low_noise_eval = False
        except:
            low_noise_eval = None
            pass
        batch["obs"] = ObsUtils.process_obs_dict(batch["obs"])
        info = PolicyAlgo.train_on_batch(self.model, batch, self.current_epoch, validate=True)
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model._forward_training(batch)
        losses = self.model._compute_losses(predictions, batch)
        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_valid.append(self.model.log_info(info))
        if self.global_step % self.model.epoch_every_n_steps == 0:
            # flatten and take the mean of the metrics
            log = {}
            for i in range(len(self.step_log_all_valid)):
                for k in self.step_log_all_valid[i]:
                    if k not in log:
                        log[k] = []
                    log[k].append(self.step_log_all_valid[i][k])
            log_all = dict((k, float(np.mean(v))) for k, v in log.items())
            for k, v in log_all.items():
                self.log("Valid/" + k, v, sync_dist=True)
            self.step_log_all_valid = []
        try:
            self.model.nets["policy"].low_noise_eval = low_noise_eval
        except:
            pass
        return losses["action_loss"]

    def configure_optimizers(self):
        if self.model.lr_schedulers["policy"]:
            lr_scheduler_dict = {
                "scheduler": self.model.lr_schedulers["policy"],
                "interval": "step",
                "frequency": 1,
            }
            return {
                "optimizer": self.model.optimizers["policy"],
                "lr_scheduler": lr_scheduler_dict,
            }
        else:
            return {
                "optimizer": self.model.optimizers["policy"],
            }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad()

    def training_epoch_end(self, outputs):
        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / int(1e9)
        self.log("System/RAM Usage (GB)", mem_usage, sync_dist=True)
        print("\nEpoch {} Memory Usage: {} GB\n".format(self.current_epoch, mem_usage))
        return super().training_epoch_end(outputs)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.model.lr_warmup:
            # lr warmup schedule taken from Gato paper
            # update params
            initial_lr = 0
            target_lr = self.model.optim_params["policy"]["learning_rate"]["initial"]
            # manually warm up lr without a scheduler
            schedule_iterations = 10000
            if self.global_step < schedule_iterations:
                for pg in self.optimizers().param_groups:
                    pg["lr"] = (
                        initial_lr
                        + (target_lr - initial_lr) * self.global_step / schedule_iterations
                    )
            else:
                scheduler.step(self.global_step - schedule_iterations)
        else:
            scheduler.step(self.global_step)


def train(config, ckpt_path, resume_dir):
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.train.seed, workers=True)
    """
    Train a model using the algorithm.
    """

    if ckpt_path is not None:
        ext_cfg = json.load(open(os.path.join(resume_dir, "config.json"), "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
        log_dir, ckpt_dir, video_dir, rollout_dir, exp_log_dir = (
            os.path.join(resume_dir, "logs"),
            os.path.join(resume_dir, "models"),
            os.path.join(resume_dir, "videos"),
            os.path.join(resume_dir, "rollouts"),
            os.path.join(resume_dir, "exp_logs"),
        )
        config.lock()
    else:
        print("\n============= New Training Run with Config =============")
        print(config)
        print("")
        log_dir, ckpt_dir, video_dir, time_str = TrainUtils.get_exp_dir(config)
        base_output_dir = os.path.join(config.train.output_dir, config.experiment.name)
        rollout_dir = os.path.join(base_output_dir, time_str, "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        exp_log_dir = os.path.join(base_output_dir, time_str, "exp_logs")
        os.makedirs(exp_log_dir, exist_ok=True)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if type(env_meta) is list:
        env_metas = env_meta
    else:
        env_metas = [env_meta]
    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        env_names_seen = []
        # create environments for validation runs
        for env_meta in env_metas:
            env_name = env_meta["env_name"]
            # check that the validation set of this env is not 0
            num_valid_demos = TrainUtils.get_num_valid_demos(config, env_name)
            if num_valid_demos == 0:
                print(
                    "No valid demos found for environment {} in dataset {}".format(
                        env_name, dataset_path
                    )
                )
                continue

            if env_name in env_names_seen:
                continue
            env_names_seen.append(env_name)
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )

            if config.train.frame_stack > 1:
                env = FrameStackWrapper(
                    env,
                    config.train.frame_stack,
                    config.experiment.rollout.horizon,
                    dataset_path=config.train.data,
                    valid_key=config.experiment.rollout.valid_key,
                )
            else:
                env = EvaluateOnDatasetWrapper(
                    env,
                    dataset_path=config.train.data,
                    valid_key=config.experiment.rollout.valid_key,
                )
            envs[env_name] = env
    print("")
    # setup for a new training runs
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=torch.device("cpu"),  # default to cpu, pl will move to gpu
    )

    if config.train.ckpt_path is not None:
        model = ModelWrapper.load_from_checkpoint(config.train.ckpt_path, model=model).model

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    loggers = [
        WandbLogger(
            project=config.wandb_project_name,
            sync_tensorboard=True,
            name=config.experiment.name,
            config=config,
            save_dir=log_dir,
        ),
    ]
    callbacks = [
        RolloutCallback(
            envs=envs,
            video_dir=video_dir,
            obs_normalization_stats=obs_normalization_stats,
            trainset=trainset,
            ckpt_dir=ckpt_dir,
            vis_data=trainset.vis_data,
            config_path=os.path.join(log_dir, "..", "config.json"),
            rollout_dir=rollout_dir,
            exp_log_dir=exp_log_dir,
        ),
        Timer(),
    ]
    if config.train.use_swa:
        callbacks.append(
            StochasticWeightAveraging(swa_lrs=config.algo.optim_params.policy.learning_rate.initial)
        )
    trainer = Trainer(
        max_steps=config.train.num_epochs * config.experiment.epoch_every_n_steps,
        accelerator="gpu",
        devices=config.train.num_gpus,
        logger=loggers,
        default_root_dir=ckpt_dir,
        callbacks=callbacks,
        fast_dev_run=config.train.fast_dev_run,
        val_check_interval=config.experiment.validation_epoch_every_n_steps,
        check_val_every_n_epoch=None,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.train.max_grad_norm,
        precision=16 if config.train.amp_enabled else 32,
        reload_dataloaders_every_n_epochs=1 if config.algo.dagger.enabled else 0,
        replace_sampler_ddp=True,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        if config.train.num_gpus > 1
        else None,
        profiler=AdvancedProfiler(dirpath=".", filename="perf_logs")
        if args.profiler != "none"
        else None,
    )

    train_sampler = trainset.get_dataset_sampler()
    valid_sampler = validset.get_dataset_sampler()

    trainer.fit(
        model=ModelWrapper(model),
        datamodule=DataModuleWrapper(
            train_dataset=trainset,
            valid_dataset=validset,
            train_dataloader_params=dict(
                sampler=train_sampler,
                batch_size=config.train.batch_size,
                shuffle=(train_sampler is None),
                num_workers=config.train.num_data_workers,
                drop_last=True,
                pin_memory=True,
            ),
            valid_dataloader_params=dict(
                sampler=valid_sampler,
                batch_size=config.train.batch_size,
                shuffle=False,
                num_workers=config.train.num_data_workers,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        ckpt_path=ckpt_path,
    )


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.output_dir is not None:
        config.train.output_dir = args.output_dir

    if args.name is not None:
        config.experiment.name = args.name

    if args.seed is not None:
        config.train.seed = args.seed
    config.train.num_gpus = args.num_gpus
    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 1 gradient steps, for 2 epochs
        config.train.fast_dev_run = 2

        # if rollouts are enabled, try 10 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 5
        config.experiment.rollout.warmstart = -1
        config.experiment.epoch_every_n_steps = 1

        # send output to a temporary directory
        config.wandb_project_name = "test"
        config.experiment.name = "test"
    elif args.profiler != "none":
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        config.experiment.epoch_every_n_steps = 10
        config.train.num_epochs = 1
        config.train.num_data_workers = 0

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        # config.experiment.rollout.rate = 1
        # config.experiment.rollout.n = 1

        # send output to a temporary directory
        config.wandb_project_name = "test"
        config.experiment.name = "test"
    else:
        config.wandb_project_name = args.wandb_project_name
        config.train.fast_dev_run = False

    if config.train.num_gpus == 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = train(config, args.ckpt_path, args.resume_dir)
        important_stats = json.dumps(important_stats, indent=4)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        print("\nRollout Success Rate Stats")
        print(important_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="(optional) if provided, override the output path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="set this flag to run a quick training run for debugging purposes",
    )

    # env seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) if provided, sets the seed",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="optimus_transformer",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to pytorch lightning ckpt file",
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="path to pytorch lightning resume dir",
    )

    parser.add_argument(
        "--profiler",
        type=str,
        default="none",
        help="profiler to use (none, pytorch, simple, advanced)",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
