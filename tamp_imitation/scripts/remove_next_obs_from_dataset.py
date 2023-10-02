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


def write_trajectory_to_dataset(env, traj, data_grp, demo_name):
    """
    Write the collected trajectory to hdf5 compatible with robomimic.
    """

    # create group for this trajectory
    ep_data_grp = data_grp.create_group(demo_name)
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))

    if "obs" in traj:
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))

    # episode metadata
    ep_data_grp.attrs["num_samples"] = traj["attrs"][
        "num_samples"
    ]  # number of transitions in this episode
    # ep_data_grp.attrs["model_file"] = traj["model_file"]
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
    for ep in demo_list:
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
        # data['model_file'] = hdf5_file["data/{}/model_file".format(ep)][()]
        write_trajectory_to_dataset(None, data, data_grp, demo_name=ep)
    env_args = hdf5_file["data/"].attrs["env_args"]

    return env_args


def combine_hdf5(hdf5_paths, hdf5_use_swmr, dataset_path):
    data_writer = h5py.File(dataset_path, "w")
    data_grp = data_writer.create_group("data")
    total_samples = 0
    dataset_keys = ["actions", "states", "rewards", "dones"]
    for hdf5_path in hdf5_paths:
        hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
        demo_list = load_demo_info(hdf5_file)
        env_args = load_dataset_in_memory(demo_list, hdf5_file, dataset_keys, data_grp)

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = env_args
    data_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--output_path")

    args = parser.parse_args()
    combine_hdf5(args.hdf5_paths, True, args.output_path)
