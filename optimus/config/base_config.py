# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""
The base config class that is used for all algorithm configs in this repository.
Subclasses get registered into a global dictionary, making it easy to instantiate
the correct config class given the algorithm name.
"""

from copy import deepcopy

import robomimic
import six  # preserve metaclass compatibility between python 2 and 3
from robomimic.config import base_config
from robomimic.config.config import Config

# global dictionary for remembering name - class mappings
REGISTERED_CONFIGS = base_config.REGISTERED_CONFIGS


def get_all_registered_configs():
    """
    Give access to dictionary of all registered configs for external use.
    """
    return deepcopy(REGISTERED_CONFIGS)


def config_factory(algo_name, dic=None):
    """
    Creates an instance of a config from the algo name. Optionally pass
    a dictionary to instantiate the config from the dictionary.
    """
    if algo_name not in REGISTERED_CONFIGS:
        raise Exception(
            "Config for algo name {} not found. Make sure it is a registered config among: {}".format(
                algo_name, ", ".join(REGISTERED_CONFIGS)
            )
        )
    return REGISTERED_CONFIGS[algo_name](dict_to_load=dic)
