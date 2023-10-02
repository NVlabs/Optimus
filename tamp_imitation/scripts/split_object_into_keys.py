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


def split_object_into_keys(all_data, ep, obs_key):
    obj = all_data[ep]["obs_key"]["object"]
    obj1_pos, obj1_quat, obj1_rel_pos, obj1_rel_quat = (
        obj[:, 0:3],
        obj[:, 3:7],
        obj[:, 7:10],
        obj[:, 10:14],
    )
    obj2_pos, obj2_quat, obj2_rel_pos, obj2_rel_quat = (
        obj[:, 14:17],
        obj[:, 17:21],
        obj[:, 21:24],
        obj[:, 24:28],
    )
    obj1_obj2_rel_pos, obj1_obj2_rel_quat = obj[:, 28:31], obj[:, 31:35]
    del all_data[ep][obs_key]["object"]
    all_data[ep][obs_key]["obj1_pos"] = obj1_pos
    all_data[ep][obs_key]["obj1_quat"] = obj1_quat
    all_data[ep][obs_key]["obj1_rel_pos"] = obj1_rel_pos
    all_data[ep][obs_key]["obj1_rel_quat"] = obj1_rel_quat
    all_data[ep][obs_key]["obj2_pos"] = obj2_pos
    all_data[ep][obs_key]["obj2_quat"] = obj2_quat
    all_data[ep][obs_key]["obj2_rel_pos"] = obj2_rel_pos
    all_data[ep][obs_key]["obj2_rel_quat"] = obj2_rel_quat
    all_data[ep][obs_key]["obj1_obj2_rel_pos"] = obj1_obj2_rel_pos
    all_data[ep][obs_key]["obj1_obj2_rel_quat"] = obj1_obj2_rel_quat


def load_dataset_in_memory(demo_list, hdf5_file, dataset_keys):
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
    all_data = dict()
    for ep in demo_list:
        all_data[ep] = {}
        all_data[ep]["attrs"] = {}
        all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        # get obs
        all_data[ep]["obs"] = {
            k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype("float32")
            for k in hdf5_file["data/{}/obs".format(ep)]
        }
        all_data[ep]["next_obs"] = {
            k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype("float32")
            for k in hdf5_file["data/{}/next_obs".format(ep)]
        }

        split_object_into_keys(all_data, ep, "obs")
        split_object_into_keys(all_data, ep, "next_obs")

        # get other dataset keys
        for k in dataset_keys:
            if k == "actions":
                all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
            else:
                all_data[ep][k] = {
                    k_: hdf5_file["data/{}/{}/k_".format(ep, k, k_)][()].astype("float32")
                    for k_ in hdf5_file["data/{}/{}".format(ep, k)].attrs
                }
        all_data[ep]["states"] = {}
        state_grp = hdf5_file["data/{}/states".format(ep)]
        for k in state_grp.keys():
            all_data[ep]["states"][k] = state_grp[k][()].astype("float32")
    env_args = hdf5_file["data/"].attrs["env_args"]
    return all_data, env_args


def combine_hdf5(hdf5_paths, hdf5_use_swmr, dataset_path):
    data_writer = h5py.File(dataset_path, "w")
    data_grp = data_writer.create_group("data")
    total_samples = 0
    dataset_keys = ["actions", "states"]
    demo_count = 0
    for hdf5_path in hdf5_paths:
        hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
        demo_list = load_demo_info(hdf5_file)
        trajs, env_args = load_dataset_in_memory(demo_list, hdf5_file, dataset_keys)
        for traj in trajs.values():
            demo_name = f"demo_{demo_count}"
            num_samples = write_trajectory_to_dataset(None, traj, data_grp, demo_name=demo_name)
            demo_count += 1
            total_samples += num_samples

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
