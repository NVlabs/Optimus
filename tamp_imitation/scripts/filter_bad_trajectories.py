# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os

import h5py
import numpy as np
from tqdm import tqdm

from tamp_imitation.scripts.combine_hdf5 import write_trajectory_to_dataset


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
    trajectories_removed = []
    for ep in tqdm(demo_list):
        data = {}
        data["attrs"] = {}
        data["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        # get obs
        data["obs"] = {
            k: hdf5_file["data/{}/obs/{}".format(ep, k)][()]
            for k in hdf5_file["data/{}/obs".format(ep)]
        }
        max_x = np.max(np.abs(data["obs"]["robot0_eef_pos"][:, 0]))
        max_y = np.max(np.abs(data["obs"]["robot0_eef_pos"][:, 1]))
        max_z = np.max(np.abs(data["obs"]["robot0_eef_pos"][:, 2]))
        if max_x > 0.2 or max_y > 0.2 or max_z > 1.1:
            print("removing {}".format(ep))
            trajectories_removed.append(ep)
            continue
        # get other dataset keys
        for k in dataset_keys:
            data[k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
        try:
            data["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]
        except:
            pass

        try:
            data["init_string"] = hdf5_file["data/{}".format(ep)].attrs["init_string"]
            data["goal_parts_string"] = hdf5_file["data/{}".format(ep)].attrs["goal_parts_string"]
        except:
            pass
        write_trajectory_to_dataset(None, data, data_grp, demo_name=ep)
    env_args = hdf5_file["data/"].attrs["env_args"]
    print("Num trajectories removed: {}".format(len(trajectories_removed)))
    print(
        "Percentage of trajectories kept: {}".format(1 - len(trajectories_removed) / len(demo_list))
    )
    return env_args


def write_hdf5s(hdf5_paths, hdf5_use_swmr, treat_as_base_path):
    total_samples = 0
    dataset_keys = ["actions", "states"]
    for hdf5_path in tqdm(hdf5_paths):
        hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
        if treat_as_base_path:
            base_path = os.path.dirname(hdf5_path)
            name = os.path.basename(hdf5_path)
            data_path = os.path.join(base_path, "filtered/", name)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        else:
            data_path = hdf5_path[: -len(".hdf5")] + "_filtered.hdf5"
        data_writer = h5py.File(data_path, "w")
        data_grp = data_writer.create_group("data")
        demo_list = load_demo_info(hdf5_file)
        env_args = load_dataset_in_memory(demo_list, hdf5_file, dataset_keys, data_grp)

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = env_args
    data_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--treat_as_base_path", action="store_true")

    args = parser.parse_args()
    write_hdf5s(args.hdf5_paths, True, args.treat_as_base_path)
