# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import copy
import json
import os
import random
import time
import traceback
from collections import OrderedDict

import imageio
import numpy as np
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.utils.train_utils import *

from tqdm import tqdm

import tamp_imitation.utils.env_utils as EnvUtils
import tamp_imitation.utils.file_utils as FileUtils
from tamp_imitation.config.base_config import config_factory
from tamp_imitation.envs.wrappers import (
    EvaluateOnDatasetWrapper,
    FrameStackWrapper,
    MultiTaskEnv,
)
from tamp_imitation.scripts.combine_hdf5 import global_dataset_updates, write_trajectory_to_dataset
from tamp_imitation.utils.dataset import SequenceDataset


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key

    # load the dataset into memory
    if config.experiment.validate:
        # assert (
        #     not config.train.hdf5_normalize_obs
        # ), "no support for observation normalization with validation data yet"
        # train_filter_by_attribute = "train"
        # valid_filter_by_attribute = "valid"
        # if filter_by_attribute is not None:
        #     train_filter_by_attribute = "{}_{}".format(
        #         filter_by_attribute, train_filter_by_attribute
        #     )
        #     valid_filter_by_attribute = "{}_{}".format(
        #         filter_by_attribute, valid_filter_by_attribute
        #     )
        # train_dataset = dataset_factory(
        #     config, obs_keys, filter_by_attribute=train_filter_by_attribute
        # )
        # valid_dataset = dataset_factory(
        #     config, obs_keys, filter_by_attribute=valid_filter_by_attribute
        # )
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute="valid")
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=config.train.load_next_obs,  # make sure dataset returns s'
        frame_stack=config.train.frame_stack,  # no frame stacking
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        transformer_enabled=config.algo.transformer.enabled,
        condition_on_actions=config.algo.transformer.condition_on_actions,
        predict_obs=config.algo.transformer.predict_obs,
        euclidean_distance_timestep=config.algo.transformer.euclidean_distance_timestep,
        language_enabled=config.algo.transformer.language_enabled,
        language_embedding=config.algo.transformer.language_embedding,
    )
    dataset = SequenceDataset(**ds_kwargs)

    return dataset


def run_rollout(
    policy,
    env,
    horizon,
    use_goals=False,
    render=False,
    video_writer=None,
    video_skip=5,
    terminate_on_success=False,
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert (
        isinstance(env, EnvBase)
        or isinstance(env.env, EnvBase)
        or isinstance(env, FrameStackWrapper)
    )

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.0
    success = {k: False for k in env.is_success()}  # success metrics
    obs_log = {k: [v.reshape(1, -1)] for k, v in ob_dict.items() if not (k.endswith("image"))}
    traj = dict(actions=[], states=[], initial_state_dict=env.get_state())
    try:
        for step_i in range(horizon):
            state_dict = env.get_state()
            traj["states"].append(state_dict["states"])
            # get action from policy
            for k in ob_dict.keys():
                ob_dict[k] = np.expand_dims(ob_dict[k], 0)
            ac = policy(ob=ob_dict, goal=goal_dict)[0]

            # play action
            ob_dict, r, done, info = env.step(
                ac,
                # render_intermediate_video_obs=env.execute_controller_actions
                # or env.execute_simplified_actions,
            )

            for k, v in ob_dict.items():
                if not (k.endswith("image")):
                    obs_log[k].append(v.reshape(1, -1))

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                # if env.execute_controller_actions or env.execute_simplified_actions:
                #     for idx, video_img in enumerate(info["intermediate_video_obs"]):
                #         if idx % video_skip == 0:
                #             video_writer.append_data(video_img)
                # else:
                if video_count % video_skip == 0:
                    video_img = []
                    video_img.append(env.render(mode="rgb_array", height=512, width=512))
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break
        state_dict = env.get_state()
        traj["states"].append(state_dict["states"])
        if not type(env.env) is EnvRobosuiteTAMP:
            traj["states"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["states"])
        traj["actions"] = np.array([0])  # just a dummy value
        traj["attrs"] = dict(num_samples=len(traj["states"]))
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    for k, v in obs_log.items():
        obs_log[k] = np.concatenate(v, axis=0)

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    results["Observations"] = obs_log
    for k, v in info.items():
        if not (k.endswith("actions")) and not (k.endswith("obs")):
            results[k] = v

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])
    return results, traj


@torch.no_grad()
def rollout_with_stats(
    policy,
    envs,
    horizon,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    video_path=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=False,
    verbose=False,
    vis_data=None,
    rollout_dir=None,
    config=None,
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

        vis_data (dict): # dict in which each item is a list of tuples (mask, xedges, yedges) for each object

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (
        video_dir is None
    ), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = {k: video_path for k in envs}
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = {k: video_writer for k in envs}
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = {k: os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs}
        video_writers = {k: imageio.get_writer(video_paths[k], fps=20) for k in envs}

    all_plots, all_plot_names = [], []
    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print(
            "rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
                env.name,
                horizon,
                use_goals,
                num_episodes,
            )
        )
        rollout_logs = []
        num_valid_demos = get_num_valid_demos(len(envs) > 1, config, env_name)
        num_episodes = min(num_valid_demos, num_episodes)
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        obs_logs = {}
        env.sample_eval_episodes(num_episodes)
        data_writer = h5py.File(os.path.join(rollout_dir, f"rollout_{epoch}.hdf5"), "w")
        data_grp = data_writer.create_group("data")
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info, traj = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            obs_logs[ep_i] = {"obs": rollout_info["Observations"]}
            del rollout_info["Observations"]
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print(
                    "Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success)
                )
                print(json.dumps(rollout_info, sort_keys=True, indent=4))
            write_trajectory_to_dataset(
                None,
                traj,
                data_grp,
                demo_name=f"demo_{ep_i}",
                env_type="mujoco",
            )
        global_dataset_updates(data_grp, 0, json.dumps(env.serialize(), indent=4))
        data_writer.close()
        successes = [rollout_log["Success_Rate"] for rollout_log in rollout_logs]
        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict(
            (
                k,
                [
                    rollout_logs[i][k]
                    for i in range(len(rollout_logs))
                    if rollout_logs[i][k] is not None
                ],
            )
            for k in rollout_logs[0]
        )
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = (
            np.sum(rollout_logs["time"]) / 60.0
        )  # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

        # success masks:
        # if vis_data[env_name] is not None and env_name in ENV_NAME_TO_INDICES_PER_OBJECT:
        #     outs = plot_x_y_data(
        #         env_name,
        #         obs_logs,
        #         path=f"rollout",
        #         save_traj_lengths=False,
        #         bin_data=vis_data[env_name],
        #         successes=successes,
        #         save_coverage_plot=True,
        #     )
        #     plots, names = [], []
        #     for out in outs:
        #         plots.append(out[3])
        #         names.append(out[4])
        # else:
        plots, names = [], []
        all_plots.extend(plots)
        all_plot_names.extend(names)

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths, all_plots, all_plot_names


def get_config_from_path(config_path):
    ext_cfg = json.load(open(os.path.join(config_path), "r"))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    return config


def make_parallel_env(
    config_path, env_id=0, render=False, multitask=False, env_meta_file=None, shape_meta=None
):
    import os

    utils.TEMP_DIR = "/tmp/temp-{}/".format(os.getpid())
    utils.VHACD_DIR = "/tmp/vhacd-{}/".format(os.getpid())
    config = get_config_from_path(config_path)
    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_path = config.train.data
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if env_meta_file is not None:
        env_meta = json.load(open(env_meta_file, "r"))
    else:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)

    if type(env_meta) is list:
        env_meta = env_meta[env_id]
    multitask = (
        "task_id" in env_meta or "task_id" in shape_meta["all_shapes"]
    )  # only use presence of task id in env_kwargs to call task id wrapper

    if "namespace_args" in env_meta["env_kwargs"]:
        env_meta["env_kwargs"]["namespace_args"]["seed"] = random.randint(0, 1000000)
        # grab the device to move the environment to the right GPU in the multi-GPU DDP setting
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        cuda_visible = int(os.getenv("CUDA_VISIBLE_DEVICES", 0))
        if cuda_visible > 0:
            device = cuda_visible
        else:
            device = local_rank
        env_meta["env_kwargs"]["namespace_args"]["compute_device_id"] = device
        env_meta["env_kwargs"]["namespace_args"]["graphics_device_id"] = device
        env_meta["env_kwargs"]["namespace_args"]["sim_device"] = f"cuda:{device}"

    env_name = env_meta["env_name"]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=config.experiment.render_video and (not render),
        use_image_obs=shape_meta["use_images"],
    )
    # this means we are doing multi-task learning
    if multitask:
        env = MultiTaskEnv(env, env_meta.get("task_id", env_id), shape_meta["all_shapes"])
        valid_key = "valid_{}".format(env_name)
    else:
        valid_key = config.experiment.rollout.valid_key

    if config.train.frame_stack > 1:
        env = FrameStackWrapper(
            env,
            config.train.frame_stack,
            config.algo.transformer.euclidean_distance_timestep,
            config.algo.transformer.open_loop_predictions,
            config.algo.transformer.reset_context_after_primitive_exec,
            config.algo.transformer.primitive_type,
            config.experiment.rollout.horizon,
            dataset_path=config.train.data,
            valid_key=valid_key,
        )
    else:
        env = EvaluateOnDatasetWrapper(
            env,
            dataset_path=config.train.data,
            valid_key=valid_key,
        )
    return env


def stack_dict(d):
    new_d = {}
    for k in d[0].keys():
        new_d[k] = np.stack([o[k] for o in d])
    return new_d


def run_parallel_rollout(
    policy,
    env,
    horizon,
    use_goals=False,
    render=False,
    video_writer=None,
    video_skip=5,
    terminate_on_success=False,
    num_envs=1,
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, SubprocVecEnv)

    policy.start_episode()

    num_val_states = env.env_method("get_num_val_states", indices=[0])[0]
    val_indices = np.random.choice(range(num_val_states), size=num_envs, replace=False)
    for idx in range(num_envs):
        env.env_method("set_eval_episode", val_indices[idx], indices=[idx])
    ob_dict = env.reset()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.0
    success = env.env_method("is_success")
    success = [{k: False for k in s} for s in success]  # success metrics
    video_images = []
    trajs = [
        dict(actions=[], states=[], initial_state_dict=env.env_method("get_state")[i])
        for i in range(num_envs)
    ]
    try:
        for step_i in range(horizon):
            state_dicts = env.env_method("get_state")
            for state_dict, traj in zip(state_dicts, trajs):
                traj["states"].append(state_dict["states"])
            ob_dict = stack_dict(ob_dict)
            # get action from policy
            ac = policy(ob=ob_dict)

            # play action
            ob_dict, r, done, info = env.step(ac)

            # compute reward
            total_reward += r

            cur_success_metrics = env.env_method("is_success")
            cur_success_metrics = [{k: bool(s[k]) for k in s} for s in cur_success_metrics]
            for s, curr_s in zip(success, cur_success_metrics):
                for k in s:
                    s[k] = s[k] or curr_s[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    ims = env.env_method("render", "rgb_array", 512, 512)
                    video_images.append(ims)
                video_count += 1

            # break if done
            if all(done) or (terminate_on_success and all([s["task"] for s in success])):
                break
        state_dicts = env.env_method("get_state")
        for idx, traj in enumerate(trajs):
            traj["states"].append(state_dicts[idx]["states"])
            traj["states"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["states"])
            traj["actions"] = np.array([0])  # just a dummy value
            traj["attrs"] = dict(num_samples=len(traj["states"]))
    except:
        print(traceback.format_exc())
        print("WARNING: got rollout exception")

    # combine the videos into one sequential video
    if video_writer is not None:
        for env_idx in range(num_envs):
            for ims in video_images:
                video_writer.append_data(ims[env_idx])

    results["Return"] = np.mean(total_reward)
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = np.mean([float(s["task"]) for s in success])
    print("Success Rate: ", results["Success_Rate"])

    return results, trajs


@torch.no_grad()
def single_env_rollout_with_stats(
    policy,
    config_path,
    env_meta,
    env_id,
    horizon,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    video_path=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=False,
    verbose=False,
    all_rollout_logs=None,
    multitask=False,
    rollout_dir=None,
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

        vis_data (dict): # dict in which each item is a list of tuples (mask, xedges, yedges) for each object

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    completed_rollouts = True
    try:
        config = get_config_from_path(config_path)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_name = env_meta["env_name"]
        env_meta = copy.deepcopy(env_meta)
        # modify num episodes based on how many demos exist in the validation set
        num_valid_demos = get_num_valid_demos(multitask, config, env_name)
        num_episodes = min(num_valid_demos, num_episodes)
        env_meta_file = env_meta[
            "env_meta_file"
        ]  # this is to ensure the env meta is not copied to each subprocess!
        env = SubprocVecEnv(
            [
                lambda: make_parallel_env(
                    config_path=config_path,
                    env_id=env_id,
                    multitask=multitask,
                    env_meta_file=env_meta_file,
                )
                for _ in range(num_episodes)
            ],
            start_method="forkserver",
        )

        # handle paths and create writers for video writing
        assert (video_path is None) or (
            video_dir is None
        ), "rollout_with_stats: can't specify both video path and dir"
        write_video = (video_path is not None) or (video_dir is not None)
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_path = os.path.join(video_dir, "{}{}".format(env_name, video_str))
        video_writer = imageio.get_writer(video_path, fps=20)
        env_video_writer = None
        if write_video:
            print("video writes to " + video_path)
            env_video_writer = video_writer

        print(
            "rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
                env_name,
                horizon,
                use_goals,
                num_episodes,
            )
        )
        rollout_timestamp = time.time()
        if rollout_dir is not None:
            data_writer = h5py.File(
                os.path.join(rollout_dir, f"rollout_{env_name}_{epoch}.hdf5"), "w"
            )
            data_grp = data_writer.create_group("data")
        rollout_logs, trajs = run_parallel_rollout(
            policy=policy,
            env=env,
            horizon=horizon,
            render=render,
            use_goals=use_goals,
            video_writer=env_video_writer,
            video_skip=video_skip,
            terminate_on_success=terminate_on_success,
            num_envs=num_episodes,
        )
        for ep_i, traj in enumerate(trajs):
            if rollout_dir is not None:
                write_trajectory_to_dataset(
                    None,
                    traj,
                    data_grp,
                    demo_name=f"demo_{ep_i}",
                    env_type="mujoco",
                )
        if rollout_dir is not None:
            global_dataset_updates(
                data_grp, 0, json.dumps(env.env_method("serialize")[0], indent=4)
            )
            data_writer.close()
        rollout_logs["Time_Episode"] = (time.time() - rollout_timestamp) / num_episodes / 60.0
        if verbose:
            print(
                "env_name= {}, horizon={}, success_rate={}".format(
                    env_name, horizon, rollout_logs["Success_Rate"]
                )
            )
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        all_rollout_logs[env_name] = rollout_logs
        all_rollout_logs[env_name + "_video_path"] = video_path
    except:
        print(traceback.format_exc())
        completed_rollouts = False
    return completed_rollouts


@torch.no_grad()
def parallel_rollout_with_stats(
    policy,
    config_path,
    envs,
    horizon,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    video_path=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=False,
    verbose=False,
    rollout_dir=None,
    multitask=False,
):
    all_rollout_logs = {}
    video_paths = {}
    for env_id, (env_name, env_meta) in tqdm(enumerate(envs.items())):
        completed_rollouts = single_env_rollout_with_stats(
            policy,
            config_path,
            env_meta,
            env_id,
            horizon,
            use_goals,
            num_episodes,
            render,
            video_dir,
            video_path,
            epoch,
            video_skip,
            terminate_on_success,
            verbose,
            all_rollout_logs,
            multitask,
            rollout_dir,
        )
        if completed_rollouts:
            video_paths[env_meta["env_name"]] = all_rollout_logs[
                env_meta["env_name"] + "_video_path"
            ]
            del all_rollout_logs[env_meta["env_name"] + "_video_path"]
    return all_rollout_logs, video_paths, [], []


def get_num_valid_demos(multitask, config, env_name):
    if multitask:
        valid_key = "valid_{}".format(env_name)
    else:
        valid_key = config.experiment.rollout.valid_key
    dataset_path = config.train.data
    if dataset_path is not None:
        hdf5_file = h5py.File(dataset_path, "r", swmr=True, libver="latest")
        filter_key = valid_key
        demos = [
            elem.decode("utf-8") for elem in np.array(hdf5_file["mask/{}".format(filter_key)][:])
        ]
        return len(demos)
    else:
        return 0
