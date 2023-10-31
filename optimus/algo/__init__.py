# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from robomimic.algo.algo import (
    Algo,
    HierarchicalAlgo,
    PlannerAlgo,
    PolicyAlgo,
    RolloutPolicy,
    ValueAlgo,
    algo_factory,
    algo_name_to_factory_func,
    register_algo_factory_func,
)

# note: these imports are needed to register these classes in the global algo registry
from robomimic.algo.bc import BC, BC_GMM, BC_VAE, BC_Gaussian
from robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from robomimic.algo.cql import CQL
from robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from robomimic.algo.hbc import HBC
from robomimic.algo.iris import IRIS
from robomimic.algo.td3_bc import TD3_BC

from optimus.algo.bc import BC_RNN, BC_RNN_GMM, BC_Transformer, BC_Transformer_GMM
