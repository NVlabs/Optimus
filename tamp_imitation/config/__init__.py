from robomimic.config.config import Config

from tamp_imitation.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from tamp_imitation.config.bc_config import BCConfig
