# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""
import os
from collections import OrderedDict

import h5py
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import *

from optimus.algo import algo_factory

import os
import shutil
import tempfile
import gdown

def download_url_from_gdrive(url, download_dir, check_overwrite=True):
    """
    Downloads a file at a URL from Google Drive.

    Example usage:
        url = https://drive.google.com/file/d/1DABdqnBri6-l9UitjQV53uOq_84Dx7Xt/view?usp=drive_link
        download_dir = "/tmp"
        download_url_from_gdrive(url, download_dir, check_overwrite=True)

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """
    assert url_is_alive(url), "@download_url_from_gdrive got unreachable url: {}".format(url)

    with tempfile.TemporaryDirectory() as td:
        # HACK: Change directory to temp dir, download file there, and then move the file to desired directory.
        #       We do this because we do not know the name of the file beforehand.
        cur_dir = os.getcwd()
        os.chdir(td)
        fpath = gdown.download(url, quiet=False, fuzzy=True)
        fname = os.path.basename(fpath)
        file_to_write = os.path.join(download_dir, fname)
        if check_overwrite and os.path.exists(file_to_write):
            user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
            assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."
        shutil.move(fpath, file_to_write)
        os.chdir(cur_dir)


def policy_from_checkpoint(device=None, ckpt_path=None, ckpt_dict=None, verbose=False):
    """
    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose)

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # env meta from model dict to get info needed to create model
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    if verbose:
        print("============= Loaded Policy =============")
        print(model)
    return model, ckpt_dict


def get_shape_metadata_from_dataset(dataset_path, all_obs_keys=None, verbose=False):
    """
    Retrieves shape metadata from dataset.

    Args:
        dataset_path (str): path to dataset
        all_obs_keys (list): list of all modalities used by the model. If not provided, all modalities
            present in the file are used.
        verbose (bool): if True, include print statements

    Returns:
        shape_meta (dict): shape metadata. Contains the following keys:

            :`'ac_dim'`: action space dimension
            :`'all_shapes'`: dictionary that maps observation key string to shape
            :`'all_obs_keys'`: list of all observation modalities used
            :`'use_images'`: bool, whether or not image modalities are present
    """

    shape_meta = {}

    # read demo file for some metadata
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    demo_id = list(f["data"].keys())[0]
    demo = f["data/{}".format(demo_id)]

    # action dimension
    shape_meta["ac_dim"] = f["data/{}/actions".format(demo_id)].shape[1]

    # observation dimensions
    all_shapes = OrderedDict()

    if all_obs_keys is None:
        # use all modalities present in the file
        all_obs_keys = [k for k in demo["obs"]]

    for k in sorted(all_obs_keys):
        if k == "timesteps":
            initial_shape = (1,)
        else:
            initial_shape = demo["obs/{}".format(k)].shape[1:]
        if verbose:
            print("obs key {} with shape {}".format(k, initial_shape))
        # Store processed shape for each obs key
        all_shapes[k] = ObsUtils.get_processed_shape(
            obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[k],
            input_shape=initial_shape,
        )

    f.close()

    shape_meta["all_shapes"] = all_shapes
    shape_meta["all_obs_keys"] = all_obs_keys
    shape_meta["use_images"] = ObsUtils.has_modality("rgb", all_obs_keys)

    return shape_meta
