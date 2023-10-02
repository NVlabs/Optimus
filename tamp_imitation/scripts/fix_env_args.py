# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
"""
import json
import os
import traceback

import h5py
import numpy as np
from tqdm import tqdm


def write_trajectory_to_dataset(
    env, traj, data_grp, demo_name, save_next_obs=False, env_type="mujoco"
):
    """
    Write the collected trajectory to hdf5 compatible with robomimic.
    """

    # create group for this trajectory
    ep_data_grp = data_grp.create_group(demo_name)
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]), compression="gzip")

    if env_type == "mujoco":
        data = np.array(traj["states"])
        ep_data_grp.create_dataset("states", data=data)
    if "obs" in traj:
        for k in traj["obs"]:
            ep_data_grp.create_dataset(
                "obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip"
            )
            if save_next_obs:
                ep_data_grp.create_dataset(
                    "next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip"
                )

    # episode metadata
    ep_data_grp.attrs["num_samples"] = traj["attrs"][
        "num_samples"
    ]  # number of transitions in this episode
    if "model_file" in traj:
        ep_data_grp.attrs["model_file"] = traj["model_file"]
    if "init_string" in traj:
        ep_data_grp.attrs["init_string"] = traj["init_string"]
    if "goal_parts_string" in traj:
        ep_data_grp.attrs["goal_parts_string"] = traj["goal_parts_string"]
    return traj["actions"].shape[0]


def load_demo_info(hdf5_file):
    """
    Args:
        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        demos (list): list of demonstration keys to load from the hdf5 file. If
            omitted, all demos in the file (or under the @filter_by_attribute
            filter key) are used.
    """
    demos = list(hdf5_file["data"].keys())

    # sort demo keys
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    return demos


def load_dataset_in_memory(
    demo_list, hdf5_file, dataset_keys, data_grp, demo_count=0, total_samples=0, env_type="mujoco"
):
    """
    Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
    differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
    `getitem` operation.

    Args:
        demo_list (list): list of demo keys, e.g., 'demo_0'
        hdf5_file (h5py.File): file handle to the hdf5 dataset.
        obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
        dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
        load_next_obs (bool): whether to load next_obs from the dataset

    Returns:
        all_data (dict): dictionary of loaded data.
    """
    env_args = hdf5_file["data/"].attrs["env_args"]
    return env_args, demo_count, total_samples


def global_dataset_updates(data_grp, total_samples, env_args):
    """
    Update the global dataset attributes.
    """
    data_grp.attrs["total_samples"] = total_samples

    return data_grp


def combine_hdf5(hdf5_paths, hdf5_use_swmr, task_name):
    for hdf5_path in tqdm(hdf5_paths):
        try:
            hdf5_file = h5py.File(hdf5_path, "r+", swmr=hdf5_use_swmr, libver="latest")
            env_args = json.loads(hdf5_file["data"].attrs["env_args"])

            asset_name = os.path.basename(hdf5_path).split("_")[0] + ".obj"
            env_args["env_kwargs"]["namespace_args"]["stamp_kwargs"] = dict(
                object_asset_path=[dataset_base + "//shapenet_sem/models/" + asset_name]
            )
            print(env_args["env_kwargs"]["namespace_args"]["stamp_kwargs"])
            if task_name:
                env_args["env_kwargs"]["namespace_args"]["task_name"] = task_name
            hdf5_file["data/"].attrs["env_args"] = json.dumps(env_args, indent=4)
            hdf5_file.close()
            # print("loaded: ", hdf5_path)
        except:
            print("failed to load: ", hdf5_path)
            print(traceback.format_exc())
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--task_name", type=str, default=None)

    args = parser.parse_args()
    combine_hdf5(args.hdf5_paths, True, args.task_name)
