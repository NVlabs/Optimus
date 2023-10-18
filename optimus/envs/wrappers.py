# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
A collection of useful environment wrappers. Taken from Ajay Mandlekar's private version of robomimic.
"""
import textwrap
from collections import deque

import h5py
import numpy as np


class Wrapper(object):
    """
    Base class for all environment wrappers in optimus.
    """

    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__)
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the
        wrapper (such as arguments passed to it).
        """
        return ""

    def __repr__(self):
        """Pretty print environment."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + "(" + msg + "\n)"
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class EvaluateOnDatasetWrapper(Wrapper):
    def __init__(
        self,
        env,
        dataset_path=None,
        valid_key="valid",
    ):
        super(EvaluateOnDatasetWrapper, self).__init__(env=env)
        self.dataset_path = dataset_path
        self.valid_key = valid_key
        if dataset_path is not None:
            self.load_evaluation_data(dataset_path)

    def sample_eval_episodes(self, num_episodes):
        """
        Sample a random set of episodes from the set of all episodes.
        """
        self.eval_indices = np.random.choice(
            range(len(self.initial_states)), size=num_episodes, replace=False
        )
        self.eval_current_index = 0

    def get_num_val_states(self):
        return len(self.demos)

    def set_eval_episode(self, eval_index):
        self.eval_indices = [eval_index]
        self.eval_current_index = 0

    def load_evaluation_data(self, hdf5_path):
        # NOTE: for the hierarchical primitive setting, this code will only
        # work for resetting the initial state, the signature/command_indices are wrong
        # DO NOT use with command_index or signature conditioning
        self.hdf5_file = h5py.File(hdf5_path, "r", swmr=True, libver="latest")
        filter_key = self.valid_key
        self.demos = [
            elem.decode("utf-8")
            for elem in np.array(self.hdf5_file["mask/{}".format(filter_key)][:])
        ]
        try:
            self.initial_states = [
                dict(
                    states=self.hdf5_file["data/{}/states".format(ep)][()][0],
                    model=self.hdf5_file["data/{}".format(ep)].attrs["model_file"],
                )
                for ep in self.demos
            ]
            self.actions = [self.hdf5_file["data/{}/actions".format(ep)][()] for ep in self.demos]
            self.obs = [
                {
                    k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()]
                    for k in ["robot0_eef_pos", "robot0_eef_quat"]
                }
                for ep in self.demos
            ]
        except:
            self.initial_states = [
                dict(
                    states={
                        k_: self.hdf5_file["data/{}/{}/{}".format(ep, "states", k_)][()].astype(
                            "float32"
                        )[0]
                        for k_ in self.hdf5_file["data/{}/{}".format(ep, "states")].keys()
                    }
                )
                for ep in self.demos
            ]
        try:
            self.command_indices = [
                self.hdf5_file["data/{}/obs/command_index".format(ep)][()] for ep in self.demos
            ]
        except:
            self.command_indices = [np.zeros(1).reshape(-1, 1) for ep in self.demos]
        try:
            self.init_strings = [
                self.hdf5_file["data/{}".format(ep)].attrs["init_string"] for ep in self.demos
            ]
            self.goal_parts_strings = [
                self.hdf5_file["data/{}".format(ep)].attrs["goal_parts_string"] for ep in self.demos
            ]
        except:
            print(traceback.format_exc())
            self.init_strings = [None for ep in self.demos]
            self.goal_parts_strings = [None for ep in self.demos]

    def reset(self):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        if self.dataset_path is not None:
            print("resetting to a valid state")
            self.env.reset()
            states = self.initial_states[self.eval_indices[self.eval_current_index]]
            if (
                self.init_strings[self.eval_indices[self.eval_current_index]] is not None
                and self.goal_parts_strings[self.eval_indices[self.eval_current_index]] is not None
            ):
                states["init_string"] = self.init_strings[
                    self.eval_indices[self.eval_current_index]
                ]
                states["goal_parts_string"] = self.goal_parts_strings[
                    self.eval_indices[self.eval_current_index]
                ]
            self.eval_current_index += 1
            obs = self.reset_to(states)
            return obs
        else:
            obs = self.env.reset()
            self.timestep = 0  # always zero regardless of timestep type
            return obs

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)


class FrameStackWrapper(Wrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """

    def __init__(
        self,
        env,
        num_frames,
        horizon=None,
        dataset_path=None,
        valid_key="valid",
    ):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert (
            num_frames > 1
        ), "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(
            num_frames
        )

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None
        self.horizon = horizon
        self.dataset_path = dataset_path
        self.valid_key = valid_key
        if dataset_path is not None:
            self.hdf5_file = h5py.File(dataset_path, "r", swmr=True, libver="latest")
            filter_key = self.valid_key
            self.demos = [
                elem.decode("utf-8")
                for elem in np.array(self.hdf5_file["mask/{}".format(filter_key)][:])
            ]

    def load_evaluation_data(self, idx):
        ep = self.demos[idx]
        initial_states = dict(
            states=self.hdf5_file["data/{}/states".format(ep)][()][0],
            model=self.hdf5_file["data/{}".format(ep)].attrs["model_file"],
        )

        try:
            init_strings = self.hdf5_file["data/{}".format(ep)].attrs["init_string"]
            goal_parts_strings = self.hdf5_file["data/{}".format(ep)].attrs["goal_parts_string"]
        except:
            init_strings = None
            goal_parts_strings = None
        return (
            initial_states,
            init_strings,
            goal_parts_strings,
        )

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)],
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def sample_eval_episodes(self, num_episodes):
        """
        Sample a random set of episodes from the set of all episodes.
        """
        self.eval_indices = np.random.choice(
            range(len(self.demos)), size=num_episodes, replace=False
        )
        self.eval_current_index = 0

    def get_num_val_states(self):
        return len(self.demos)

    def set_eval_episode(self, eval_index):
        self.eval_indices = [eval_index]
        self.eval_current_index = 0

    def reset(self, use_eval_indices=True):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        if self.dataset_path is not None and use_eval_indices:
            print("resetting to a valid state")
            self.env.reset()
            (
                states,
                init_string,
                goal_parts_string,
            ) = self.load_evaluation_data(self.eval_indices[self.eval_current_index])
            if init_string is not None and goal_parts_string is not None:
                states["init_string"] = init_string
                states["goal_parts_string"] = goal_parts_string
            self.eval_current_index += 1
            obs = self.reset_to(states)
            return obs
        else:
            obs = self.env.reset()
            self.timestep = 0  # always zero regardless of timestep type
            self.update_obs(obs, reset=True)
            self.obs_history = self._get_initial_obs_history(init_obs=obs)
            return self._get_stacked_obs_from_history()

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action, **kwargs):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action, **kwargs)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret, r, done, info

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)
