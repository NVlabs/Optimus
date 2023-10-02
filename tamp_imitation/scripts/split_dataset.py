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

import h5py
import numpy as np
from robomimic.utils.file_utils import create_hdf5_filter_key


def split_demos_from_hdf5(hdf5_path, num_demos=None, filter_key=None):
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
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])])
    else:
        demos = sorted(list(f["data"].keys()))
    num_total_demos = len(demos)
    f.close()
    # get random split
    for num_demo in num_demos:
        num_demo = int(num_demo)
        mask = np.zeros(num_total_demos)
        mask[:num_demo] = 1.0
        np.random.shuffle(mask)
        mask = mask.astype(int)
        inds = mask.nonzero()[0]
        keys = [demos[i] for i in inds]
        print("{} demonstrations out of {} total demonstrations.".format(num_demo, num_total_demos))

        # pass mask to generate split
        if filter_key is not None:
            name = f"{filter_key}_{num_demo}_split"
        else:
            name = f"{num_demo}_split"
        lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=keys, key_name=name)
        # print(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="if provided, split the subset of trajectories in the file that correspond to\
            this filter key into a training and validation set of trajectories, instead of\
            splitting the full set of trajectories",
    )
    parser.add_argument("--num_demos", nargs="+")
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    split_demos_from_hdf5(args.dataset, num_demos=args.num_demos, filter_key=args.filter_key)
