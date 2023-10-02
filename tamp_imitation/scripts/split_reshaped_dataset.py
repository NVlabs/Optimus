# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Script for splitting a dataset hdf5 file into training and validation trajectories.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, split the subset of trajectories
        in the file that correspond to this filter key into a training
        and validation set of trajectories, instead of splitting the
        full set of trajectories

    ratio (float): validation ratio, in (0, 1). Defaults to 0.1, which is 10%.

Example usage:
    python split_train_val.py --dataset /path/to/demo.hdf5 --ratio 0.1
"""

import argparse
from collections import OrderedDict

import h5py
import numpy as np
from robomimic.utils.file_utils import create_hdf5_filter_key


def get_per_primitive_filter_keys(demo_list, hdf5_file, primitive_type, train_keys, valid_keys):
    per_primitive_train_filter_keys = OrderedDict()
    per_primitive_valid_filter_keys = OrderedDict()

    for ep in demo_list:
        # get obs
        primitives = hdf5_file["data/{}/obs/{}".format(ep, primitive_type)][()][:, 0]
        primitive = primitives[0]
        demo_name = ep
        if ep in valid_keys:
            if primitive in per_primitive_valid_filter_keys:
                per_primitive_valid_filter_keys[primitive].append(demo_name)
            else:
                per_primitive_valid_filter_keys[primitive] = [demo_name]
        elif ep in train_keys:
            if primitive in per_primitive_train_filter_keys:
                per_primitive_train_filter_keys[primitive].append(demo_name)
            else:
                per_primitive_train_filter_keys[primitive] = [demo_name]
    return per_primitive_train_filter_keys, per_primitive_valid_filter_keys


def split_demos_from_hdf5(hdf5_path):
    """
    Splits data into training set and validation set from HDF5 file.

    Args:
        hdf5_path (str): path to the hdf5 file
            to load the transitions from

        val_ratio (float): ratio of validation demonstrations to all demonstrations

        filter_key (str): if provided, split the subset of demonstration keys stored
            under mask/@filter_key instead of the full set of demonstrations
    """

    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    demos = sorted(list(f["data"].keys()))

    # get train_0 keys:
    train_0_keys = []
    train_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format("train")][:])]

    valid_keys = [elem for elem in demos if elem not in train_keys]
    for demo in train_keys:
        if demo.endswith("0"):
            train_0_keys.append(demo)
    (
        per_primitive_train_filter_keys,
        per_primitive_valid_filter_keys,
    ) = get_per_primitive_filter_keys(demos, f, "combinatorial_stack_id", train_keys, valid_keys)
    f.close()
    create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_0_keys, key_name="train_start")
    for primitive_id in per_primitive_train_filter_keys.keys():
        primitive_train_keys = per_primitive_train_filter_keys[primitive_id]
        primitive_valid_keys = per_primitive_valid_filter_keys[primitive_id]
        create_hdf5_filter_key(
            hdf5_path=hdf5_path,
            demo_keys=primitive_train_keys,
            key_name=f"train_{primitive_id}",
        )
        create_hdf5_filter_key(
            hdf5_path=hdf5_path,
            demo_keys=primitive_valid_keys,
            key_name=f"valid_{primitive_id}",
        )
        print(primitive_id, len(primitive_train_keys), len(primitive_valid_keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()
    # seed to make sure results are consistent
    np.random.seed(0)
    split_demos_from_hdf5(args.dataset)
