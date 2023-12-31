# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
import traceback
from collections import Counter

import h5py
import numpy as np
from robomimic.utils.file_utils import create_hdf5_filter_key
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

    data = np.array(traj["states"])
    ep_data_grp.create_dataset("states", data=data)
    if "obs" in traj:
        for k in traj["obs"]:
            ep_data_grp.create_dataset(
                "obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip"
            )
            if save_next_obs:
                ep_data_grp.create_dataset(
                    "next_obs/{}".format(k),
                    data=np.array(traj["next_obs"][k]),
                    compression="gzip",
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
            to select a subset of demonstval_ration trajectories to load

        demos (list): list of demonstval_ration keys to load from the hdf5 file. If
            omitted, all demos in the file (or under the @filter_by_attribute
            filter key) are used.
    """
    demos = list(hdf5_file["data"].keys())

    # sort demo keys
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    return demos


def compute_traj_length_stats(hdf5_file, demo_list):
    traj_lengths = []
    for ep in demo_list:
        traj_lengths.append(hdf5_file["data"][ep].attrs["num_samples"])
    return np.mean(traj_lengths), np.std(traj_lengths)


def combine_hdf5(
    hdf5_paths,
    hdf5_use_swmr,
    outlier_traj_length_sd,
    x_bounds,
    y_bounds,
    z_bounds,
    val_ratio,
    filter_key_prefix,
    eef_pos_key,
):
    traj_lengths = []
    outlier_files = Counter()
    out_of_bb_files = Counter()
    num_outlier_trajectories = 0
    num_out_of_bb_trajectories = 0
    num_trajectories = 0
    for hdf5_path in tqdm(hdf5_paths):
        try:
            hdf5_file = h5py.File(hdf5_path, "r+", swmr=hdf5_use_swmr, libver="latest")
            demo_list = load_demo_info(hdf5_file)
            traj_length_mean, traj_length_std = compute_traj_length_stats(hdf5_file, demo_list)
            filtered_demos = []
            for ep in demo_list:
                eef_pos = hdf5_file["data/{}/obs/{}".format(ep, eef_pos_key)][()]
                traj_length = hdf5_file["data"][ep].attrs["num_samples"]
                if traj_length > traj_length_mean + outlier_traj_length_sd * traj_length_std:
                    too_long_traj = True
                else:
                    too_long_traj = False
                if x_bounds is not None and y_bounds is not None and z_bounds is not None:
                    out_of_bb = (
                        min(eef_pos[:, 0]) < x_bounds[0]
                        or max(eef_pos[:, 0]) > x_bounds[1]
                        or min(eef_pos[:, 1]) < y_bounds[0]
                        or max(eef_pos[:, 1]) > y_bounds[1]
                        or min(eef_pos[:, 2]) < z_bounds[0]
                        or max(eef_pos[:, 2]) > z_bounds[1]
                    )
                else:
                    out_of_bb = False
                if out_of_bb:
                    out_of_bb_files[hdf5_path] += 1
                    num_out_of_bb_trajectories += 1
                elif too_long_traj:
                    outlier_files[hdf5_path] += 1
                    num_outlier_trajectories += 1
                else:
                    filtered_demos.append(ep)
                    traj_lengths.append(hdf5_file["data"][ep].attrs["num_samples"])
            num_trajectories += len(demo_list)
            hdf5_file.close()

            num_demos = len(filtered_demos)
            val_val_ratio = val_ratio
            num_val = int(val_val_ratio * num_demos)
            mask = np.zeros(num_demos)
            mask[:num_val] = 1.0
            np.random.shuffle(mask)
            mask = mask.astype(int)
            train_inds = (1 - mask).nonzero()[0]
            valid_inds = mask.nonzero()[0]
            train_keys = [filtered_demos[i] for i in train_inds]
            valid_keys = [filtered_demos[i] for i in valid_inds]
            for key in train_keys:
                assert not (key in valid_keys)
            print(
                "{} validation demonstval_rations out of {} total demonstval_rations.".format(
                    num_val, num_demos
                )
            )

            # pass mask to generate split
            name_1 = f"train{filter_key_prefix}"
            name_2 = f"valid{filter_key_prefix}"

            train_lengths = create_hdf5_filter_key(
                hdf5_path=hdf5_path, demo_keys=train_keys, key_name=name_1
            )
            valid_lengths = create_hdf5_filter_key(
                hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=name_2
            )
            all_valid_lengths = create_hdf5_filter_key(
                hdf5_path=hdf5_path, demo_keys=filtered_demos, key_name="all_valid"
            )

            print("Total number of train samples: {}".format(np.sum(train_lengths)))
            print("Average number of train samples {}".format(np.mean(train_lengths)))

            print("Total number of valid samples: {}".format(np.sum(valid_lengths)))
            print("Average number of valid samples {}".format(np.mean(valid_lengths)))

            print("Total number of all valid samples: {}".format(np.sum(all_valid_lengths)))
            print("Average number of all valid samples {}".format(np.mean(all_valid_lengths)))
        except:
            print("failed to load: ", hdf5_path)
            print(traceback.format_exc())
            pass
    print(f"Num files with > mean + 2*sd samples: ", len(outlier_files))
    print(
        f"Num trajectories with > mean + 2*sd samples: ",
        num_outlier_trajectories,
    )
    print(
        "percentage of outlier trajectories: ",
        num_outlier_trajectories / num_trajectories * 100,
    )
    print()
    print(f"Num files with out of bb: ", len(out_of_bb_files))
    print(f"Num trajectories with out of bb: ", num_out_of_bb_trajectories)
    print(
        "percentage of out of bb trajectories: ",
        num_out_of_bb_trajectories / num_trajectories * 100,
    )

    # print trajectory stats:
    print("min traj length: ", np.min(traj_lengths))
    print("max traj length: ", np.max(traj_lengths))
    print("mean traj length: ", np.mean(traj_lengths))
    print("median traj length: ", np.median(traj_lengths))
    print("std traj length: ", np.std(traj_lengths))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--outlier_traj_length_sd", type=int, default=None)
    parser.add_argument("--x_bounds", nargs=2, type=float, default=None)
    parser.add_argument("--y_bounds", nargs=2, type=float, default=None)
    parser.add_argument("--z_bounds", nargs=2, type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--filter_key_prefix", type=str, default="")
    parser.add_argument("--eef_pos_key", type=str, default="robot0_eef_pos")

    args = parser.parse_args()
    combine_hdf5(
        args.hdf5_paths,
        True,
        args.outlier_traj_length_sd,
        args.x_bounds,
        args.y_bounds,
        args.z_bounds,
        val_ratio=args.val_ratio,
        filter_key_prefix=args.filter_key_prefix,
        eef_pos_key=args.eef_pos_key,
    )
