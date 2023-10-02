# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os

from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--ratio", type=float, default=0.05)

    args = parser.parse_args()

    for hdf5 in tqdm(os.listdir(args.data_path)):
        os.system(
            f"python3 tamp_imitation/scripts/split_train_val.py --dataset {os.path.join(args.data_path, hdf5)} --ratio {args.ratio}"
        )
