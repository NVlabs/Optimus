# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from robomimic.config.config import Config

from optimus.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from optimus.config.bc_config import BCConfig
