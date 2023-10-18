# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import numpy as np
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.dataset import SequenceDataset


class SequenceDataset(SequenceDataset):
    def __init__(
        self,
        *args,
        transformer_enabled=False,
        **kwargs,
    ):
        self.transformer_enabled = transformer_enabled
        self.vis_data = dict()
        self.ep_to_hdf5_file = None
        super().__init__(*args, **kwargs)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """
        if self.ep_to_hdf5_file is None:
            self.ep_to_hdf5_file = {ep: self.hdf5_file for ep in self.demos}
        # check if this key should be in memory
        key_should_be_in_memory = self.hdf5_cache_mode in ["all", "low_dim"]
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if "/" in key:
                key1, key2 = key.split("/")
                assert key1 in ["obs", "next_obs"]
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if "/" in key:
                key1, key2 = key.split("/")
                assert key1 in ["obs", "next_obs"]
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.ep_to_hdf5_file[ep][hd5key]
        return ret

    def get_sequence_from_demo(
        self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1
    ):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index:seq_end_index]
        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array(
            [0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad
        )
        pad_mask = pad_mask[:, None].astype(np.bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(
        self,
        demo_id,
        index_in_demo,
        keys,
        num_frames_to_stack=0,
        seq_length=1,
        prefix="obs",
    ):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple("{}/{}".format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split("/")[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        return obs

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
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

        print("SequenceDataset: loading dataset into memory...")
        obs_keys = [o for o in obs_keys if o != "timesteps" and o != "goal"]

        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs[
                "num_samples"
            ]

            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
                else:
                    all_data[ep][k] = np.zeros(
                        (all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32
                    )
            # get obs
            all_data[ep]["obs"] = {
                k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype("float32") for k in obs_keys
            }

            if self.load_next_obs:
                # last block position is given by last elem of next_obs
                goal = hdf5_file["data/{}/next_obs/{}".format(ep, "object")][()].astype("float32")[
                    -1, 7:10
                ]
                all_data[ep]["obs"]["goal"] = np.repeat(
                    goal.reshape(1, -1), all_data[ep]["attrs"]["num_samples"], axis=0
                )

            if self.transformer_enabled:
                all_data[ep]["obs"]["timesteps"] = np.arange(
                    0, all_data[ep]["obs"][obs_keys[0]].shape[0]
                ).reshape(-1, 1)
            if load_next_obs:
                all_data[ep]["next_obs"] = {
                    k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype("float32")
                    for k in obs_keys
                }
                if self.transformer_enabled:
                    # Doesn't actually matter, won't be used
                    all_data[ep]["next_obs"]["timesteps"] = np.zeros_like(
                        all_data[ep]["obs"]["timesteps"]
                    )
                all_data[ep]["next_obs"]["goal"] = np.repeat(
                    goal.reshape(1, -1), all_data[ep]["attrs"]["num_samples"], axis=0
                )
        return all_data

    def get_dataset_sequence_from_demo(
        self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1
    ):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        keys = [*self.dataset_keys]
        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs",
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(
                meta["obs"], obs_normalization_stats=self.obs_normalization_stats
            )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(
                    meta["next_obs"],
                    obs_normalization_stats=self.obs_normalization_stats,
                )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(
                    goal, obs_normalization_stats=self.obs_normalization_stats
                )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        return meta

    def update_demo_info(self, demos, online_epoch, data, hdf5_file=None):
        """
        This function is called during online epochs to update the demo information based
        on newly collected demos.
        Args:
            demos (list): list of demonstration keys to load data.
            online_epoch (int): value of the current online epoch
            data (dict): dictionary containing newly collected demos
        """
        # sort demo keys
        inds = np.argsort(
            [int(elem[5:]) for elem in demos if not (elem in ["env_args", "model_file"])]
        )
        new_demos = [demos[i] for i in inds]
        self.demos.extend(new_demos)

        self.n_demos = len(self.demos)

        self.prev_total_num_sequences = self.total_num_sequences
        for new_ep in new_demos:
            self.ep_to_hdf5_file[new_ep] = hdf5_file
            demo_length = data[new_ep]["num_samples"]
            self._demo_id_to_start_indices[new_ep] = self.total_num_sequences
            self._demo_id_to_demo_length[new_ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= self.n_frame_stack - 1
            if not self.pad_seq_length:
                num_sequences -= self.seq_length - 1

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert (
                    num_sequences >= 1
                )  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = new_ep
                self.total_num_sequences += 1
        return new_demos

    def update_dataset_in_memory(
        self,
        demo_list,
        data,
        obs_keys,
        dataset_keys,
        load_next_obs=False,
        online_epoch=0,
    ):
        """
        Loads the newly collected dataset into memory, preserving the structure of the data. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            data (dict): dictionary containing newly collected demos
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        obs_keys = [o for o in obs_keys if o != "timesteps"]
        for new_ep in LogUtils.custom_tqdm(demo_list):
            all_data[new_ep] = {}
            all_data[new_ep]["attrs"] = {}
            all_data[new_ep]["attrs"]["num_samples"] = data[new_ep]["num_samples"]

            # get other dataset keys
            for k in dataset_keys:
                if k in data[new_ep]:
                    all_data[new_ep][k] = data[new_ep][k].astype("float32")
                else:
                    all_data[new_ep][k] = np.zeros(
                        (all_data[new_ep]["attrs"]["num_samples"], 1), dtype=np.float32
                    )
            # get obs
            all_data[new_ep]["obs"] = {
                k: data[new_ep]["obs"][k] for k in obs_keys if k != "timesteps"
            }

            for k in all_data[new_ep]["obs"]:
                all_data[new_ep]["obs"][k] = all_data[new_ep]["obs"][k].astype("float32")

            if self.transformer_enabled:
                all_data[new_ep]["obs"]["timesteps"] = np.arange(
                    0, all_data[new_ep]["obs"][obs_keys[0]].shape[0]
                ).reshape(-1, 1)

        self.hdf5_cache.update(all_data)
