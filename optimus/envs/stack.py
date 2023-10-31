# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from optimus.envs.box import BoxObjectWithSites


class Stack_Optimus(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        cubeA_pos = self.sim.data.xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Aligning is successful when cubeA is right above cubeB
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
            r_stack = 2.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObjectWithSites(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObjectWithSites(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        self.cubes = cubes
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
        self.cubeB_body_id = self.sim.model.body_name2id(self.cubeB.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(self.sim.data.xpos[self.cubeB_body_id])

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.xquat[self.cubeB_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return (
                    obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeB_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeB_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [
                cubeA_pos,
                cubeA_quat,
                cubeB_pos,
                cubeB_quat,
                gripper_to_cubeA,
                gripper_to_cubeB,
                cubeA_to_cubeB,
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)


class StackThree(Stack_Optimus):
    """
    Stack three cubes instead of two.
    """

    def reward(self, action=None):
        """
        We only return sparse rewards here.
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeC_lifted(self):
        # cube C needs to be higher than A
        return self._check_lifted(self.cubeC_body_id, margin=0.08)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return cubeA_lifted and cubeA_touching_cubeB

    def _check_cubeC_stacked(self):
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        cubeC_touching_cubeA = self.check_contact(self.cubeC, self.cubeA)
        return cubeC_lifted and cubeC_touching_cubeA

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A.
        # This means both A and C are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.0
        r_lift = 0.0
        r_stack = 0.0
        if self._check_cubeA_stacked() and self._check_cubeC_stacked():
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObjectWithSites(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObjectWithSites(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObjectWithSites(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        self.cubes = cubes
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
            # for cube, object_name in zip(cubes, ["cubeA", "cubeB", "cubeC"]):
            #     self.placement_initializer.add_objects_to_sampler(sampler_name=f"{object_name}Sampler", mujoco_objects=cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            # self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            # object_names = ['cubeA', 'cubeB', 'cubeC']
            # x_ranges = [[-.1, -.025], [.025, .1], [-.1, -.025]]
            # y_ranges = [[-.1, -.025], [.025, .1], [.025, 0.1]]
            # for cube, object_name, x_range, y_range in zip(cubes, object_names, x_ranges, y_ranges):
            #     self.placement_initializer.append_sampler(
            #             sampler=UniformRandomSampler(
            #             name=f"{object_name}Sampler",
            #             x_range=x_range,
            #             y_range=y_range,
            #             rotation=None,
            #             ensure_object_boundary_in_range=False,
            #             ensure_valid_placement=True,
            #             reference_pos=self.table_offset,
            #             z_offset=0.01,
            #         )
            #     )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.xpos[self.cubeC_body_id])

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.xquat[self.cubeC_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeC(obs_cache):
                return (
                    obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeC(obs_cache):
                return (
                    obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeB_to_cubeC(obs_cache):
                return (
                    obs_cache["cubeB_pos"] - obs_cache["cubeC_pos"]
                    if "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [
                cubeC_pos,
                cubeC_quat,
                gripper_to_cubeC,
                cubeA_to_cubeC,
                cubeB_to_cubeC,
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


class StackFour(StackThree):
    """
    Stack four cubes instead of three.
    """

    def reward(self, action=None):
        """
        We only return sparse rewards here.
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _check_cubeD_lifted(self):
        # cube D needs to be higher than C
        return self._check_lifted(self.cubeD_body_id, margin=0.12)

    def _check_cubeD_stacked(self):
        grasping_cubeD = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeD)
        cubeD_lifted = self._check_cubeD_lifted()
        cubeD_touching_cubeC = self.check_contact(self.cubeD, self.cubeC)
        return cubeD_lifted and cubeD_touching_cubeC

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        r_reach = 0.0
        r_lift = 0.0
        r_stack = 0.0
        if (
            self._check_cubeA_stacked()
            and self._check_cubeC_stacked()
            and self._check_cubeD_stacked()
        ):
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cubeA = BoxObjectWithSites(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObjectWithSites(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObjectWithSites(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )

        self.cubeD = BoxObjectWithSites(
            name="cubeD",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=darkwood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]
        self.cubes = cubes
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.12, 0.12],
                y_range=[-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)
        self.cubeD_body_id = self.sim.model.body_name2id(self.cubeD.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeD_pos(obs_cache):
                return np.array(self.sim.data.xpos[self.cubeD_body_id])

            @sensor(modality=modality)
            def cubeD_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.xquat[self.cubeD_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeD(obs_cache):
                return (
                    obs_cache["cubeD_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeD_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeD(obs_cache):
                return (
                    obs_cache["cubeD_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeD_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeB_to_cubeD(obs_cache):
                return (
                    obs_cache["cubeD_pos"] - obs_cache["cubeB_pos"]
                    if "cubeB_pos" in obs_cache and "cubeD_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeC_to_cubeD(obs_cache):
                return (
                    obs_cache["cubeD_pos"] - obs_cache["cubeC_pos"]
                    if "cubeC_pos" in obs_cache and "cubeD_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [
                cubeD_pos,
                cubeD_quat,
                gripper_to_cubeD,
                cubeA_to_cubeD,
                cubeB_to_cubeD,
                cubeC_to_cubeD,
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


class StackFive(StackFour):
    """
    Stack five cubes instead of four.
    """

    def _check_cubeE_lifted(self):
        # cube E needs to be higher than D
        return self._check_lifted(self.cubeE_body_id, margin=0.16)

    def _check_cubeE_stacked(self):
        grasping_cubeE = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeE)
        cubeE_lifted = self._check_cubeE_lifted()
        cubeE_touching_cubeD = self.check_contact(self.cubeE, self.cubeD)
        return cubeE_lifted and cubeE_touching_cubeD

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A and D is on top of C.
        # This means both A, C, D are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C) and (C, D).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.0
        r_lift = 0.0
        r_stack = 0.0
        if (
            self._check_cubeA_stacked()
            and self._check_cubeC_stacked()
            and self._check_cubeD_stacked()
            and self._check_cubeE_stacked()
        ):
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cubeA = BoxObjectWithSites(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObjectWithSites(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObjectWithSites(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )

        self.cubeD = BoxObjectWithSites(
            name="cubeD",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=darkwood,
        )

        self.cubeE = BoxObjectWithSites(
            name="cubeE",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=lightwood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD, self.cubeE]
        self.cubes = cubes
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.14, 0.14],
                y_range=[-0.14, 0.14],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube E
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeE_body_id = self.sim.model.body_name2id(self.cubeE.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeE_pos(obs_cache):
                return np.array(self.sim.data.xpos[self.cubeE_body_id])

            @sensor(modality=modality)
            def cubeE_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.xquat[self.cubeE_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeE(obs_cache):
                return (
                    obs_cache["cubeE_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeE_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeA_to_cubeE(obs_cache):
                return (
                    obs_cache["cubeE_pos"] - obs_cache["cubeA_pos"]
                    if "cubeA_pos" in obs_cache and "cubeE_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeB_to_cubeE(obs_cache):
                return (
                    obs_cache["cubeE_pos"] - obs_cache["cubeB_pos"]
                    if "cubeB_pos" in obs_cache and "cubeE_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeC_to_cubeE(obs_cache):
                return (
                    obs_cache["cubeE_pos"] - obs_cache["cubeC_pos"]
                    if "cubeC_pos" in obs_cache and "cubeE_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeD_to_cubeE(obs_cache):
                return (
                    obs_cache["cubeE_pos"] - obs_cache["cubeD_pos"]
                    if "cubeD_pos" in obs_cache and "cubeE_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [
                cubeE_pos,
                cubeE_quat,
                gripper_to_cubeE,
                cubeA_to_cubeE,
                cubeB_to_cubeE,
                cubeC_to_cubeE,
                cubeD_to_cubeE,
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


class StackFourPickPlace(StackFour):
    """
    Stack four cubes on four fixed cubes.
    """

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeB_lifted(self):
        return self._check_lifted(self.cubeB_body_id, margin=0.04)

    def _check_cubeC_lifted(self):
        return self._check_lifted(self.cubeC_body_id, margin=0.04)

    def _check_cubeD_lifted(self):
        return self._check_lifted(self.cubeD_body_id, margin=0.04)

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A and D is on top of C.
        # This means both A, C, D are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C) and (C, D).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.0
        r_lift = 0.0
        r_stack = 0.0
        r_stack = 0.0  # unused

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cubeA = BoxObjectWithSites(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObjectWithSites(
            name="cubeB",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObjectWithSites(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )

        self.cubeD = BoxObjectWithSites(
            name="cubeD",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=darkwood,
        )

        self.baseA = BoxObjectWithSites(
            name="baseA",
            size_min=[0.04, 0.04, 0.01],
            size_max=[0.04, 0.04, 0.01],
            rgba=[1, 0, 0, 1],
            material=lightwood,
        )

        self.baseB = BoxObjectWithSites(
            name="baseB",
            size_min=[0.04, 0.04, 0.01],
            size_max=[0.04, 0.04, 0.01],
            rgba=[1, 0, 0, 1],
            material=lightwood,
        )

        self.baseC = BoxObjectWithSites(
            name="baseC",
            size_min=[0.04, 0.04, 0.01],
            size_max=[0.04, 0.04, 0.01],
            rgba=[1, 0, 0, 1],
            material=lightwood,
        )

        self.baseD = BoxObjectWithSites(
            name="baseD",
            size_min=[0.04, 0.04, 0.01],
            size_max=[0.04, 0.04, 0.01],
            rgba=[1, 0, 0, 1],
            material=lightwood,
        )

        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]
        bases = [self.baseA, self.baseB, self.baseC, self.baseD]
        self.cubes = cubes + bases  # TODO: change this
        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(cubes)
        # else:
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="CubesSampler",
                mujoco_objects=cubes,
                x_range=[-0.15, 0.15],
                y_range=[-0.2, -0.05],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"baseASampler",
                mujoco_objects=self.baseA,
                x_range=[-0.05, -0.05],
                y_range=[0.075, 0.075],
                rotation=np.pi,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"baseBSampler",
                mujoco_objects=self.baseB,
                x_range=[0.075, 0.075],
                y_range=[0.075, 0.075],
                rotation=np.pi,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"baseCSampler",
                mujoco_objects=self.baseC,
                x_range=[-0.05, -0.05],
                y_range=[0.2, 0.2],
                rotation=np.pi,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"baseDSampler",
                mujoco_objects=self.baseD,
                x_range=[0.075, 0.075],
                y_range=[0.2, 0.2],
                rotation=np.pi,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()
        self.placement_initializer.add_objects_to_sampler(
            sampler_name="CubesSampler", mujoco_objects=cubes
        )
        self.placement_initializer.add_objects_to_sampler(
            sampler_name="baseASampler", mujoco_objects=self.baseA
        )
        self.placement_initializer.add_objects_to_sampler(
            sampler_name="baseBSampler", mujoco_objects=self.baseB
        )
        self.placement_initializer.add_objects_to_sampler(
            sampler_name="baseCSampler", mujoco_objects=self.baseC
        )
        self.placement_initializer.add_objects_to_sampler(
            sampler_name="baseDSampler", mujoco_objects=self.baseD
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes + bases,
        )
