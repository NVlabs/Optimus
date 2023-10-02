# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import h5py
import numpy as np

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
    demos_to_remove,
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
    print(demos_to_remove)
    for ep in demo_list:
        if ep in demos_to_remove:
            print(ep)
            continue
        data = {}
        data["attrs"] = {}
        data["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        # get obs
        data["obs"] = {
            k: hdf5_file["data/{}/obs/{}".format(ep, k)][()]
            for k in hdf5_file["data/{}/obs".format(ep)]
        }
        # get other dataset keys
        for k in dataset_keys:
            data[k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
        data["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]
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
    return env_args


def combine_hdf5(hdf5_paths, hdf5_use_swmr, dataset_path, demos_to_remove):
    data_writer = h5py.File(dataset_path, "w")
    data_grp = data_writer.create_group("data")
    total_samples = 0
    dataset_keys = ["actions", "states"]
    for hdf5_path in hdf5_paths:
        hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
        demo_list = load_demo_info(hdf5_file)
        env_args = load_dataset_in_memory(
            demo_list, hdf5_file, dataset_keys, demos_to_remove, data_grp
        )

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = env_args
    data_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--output_path")
    parser.add_argument("--demos_to_remove", nargs="+")

    args = parser.parse_args()
    combine_hdf5(args.hdf5_paths, True, args.output_path, args.demos_to_remove)
