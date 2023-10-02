# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import traceback

import h5py
import numpy as np
from tqdm import tqdm

from tamp_imitation.scripts.combine_hdf5 import global_dataset_updates, write_trajectory_to_dataset


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
    demo_list,
    hdf5_file,
    dataset_keys,
    data_grp,
    demo_count=0,
    total_samples=0,
    ep_data_grp_file=None,
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
    for ep in tqdm(demo_list):
        demo_name = f"demo_{demo_count}"
        traj = {}
        traj["attrs"] = {}
        traj["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        # get obs
        traj["obs"] = {
            k: hdf5_file["data/{}/obs/{}".format(ep, k)][()]
            for k in hdf5_file["data/{}/obs".format(ep)]
        }
        # get other dataset keys
        for k in dataset_keys:
            traj[k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
        try:
            traj["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]
        except:
            pass
        traj["init_string"] = ep_data_grp_file["data/{}".format(ep)].attrs["init_string"]
        traj["goal_parts_string"] = ep_data_grp_file["data/{}".format(ep)].attrs[
            "goal_parts_string"
        ]
        write_trajectory_to_dataset(None, traj, data_grp, demo_name=demo_name)
        demo_count += 1
        total_samples += traj["attrs"]["num_samples"]
    env_args = hdf5_file["data/"].attrs["env_args"]
    return env_args, demo_count, total_samples


def combine_hdf5(ep_data_grp_file, data_file, hdf5_use_swmr, output_path):
    data_writer = h5py.File(output_path, "w")
    data_grp = data_writer.create_group("data")
    dataset_keys = ["actions", "states"]
    demo_count = 0
    total_samples = 0
    ep_data_grp_file = h5py.File(ep_data_grp_file, "r", swmr=hdf5_use_swmr, libver="latest")
    data_file = h5py.File(data_file, "r", swmr=hdf5_use_swmr, libver="latest")
    demo_list = load_demo_info(data_file)
    env_args, demo_count, total_samples = load_dataset_in_memory(
        demo_list,
        data_file,
        dataset_keys,
        data_grp=data_grp,
        total_samples=total_samples,
        demo_count=demo_count,
        ep_data_grp_file=ep_data_grp_file,
    )
    # this won't work when combining many files (just re-generate filter keys)
    # if "mask" in hdf5_file:
    #     hdf5_file.copy("mask", data_writer)
    print("loaded: ", data_file)
    global_dataset_updates(data_grp, total_samples, env_args)
    data_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_data_grp_file")
    parser.add_argument("--data_file")
    parser.add_argument("--output_path")

    args = parser.parse_args()
    combine_hdf5(args.ep_data_grp_file, args.data_file, True, args.output_path)
