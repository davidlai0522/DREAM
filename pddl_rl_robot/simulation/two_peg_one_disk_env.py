from pddl_rl_robot.simulation.peg_and_hole_base_env import NutAssembly, PegsArena
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.base import MujocoModel
from robosuite.models.objects import RoundNutObject
import numpy as np


class TwoPegOneRoundNut(NutAssembly):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        assert (
            "single_object_mode" not in kwargs and "nut_type" not in kwargs
        ), "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="round", **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        ManipulationEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # define nuts - only one round nut
        self.nuts = []
        nut_names = ("RoundNut",)  # Only include round nut

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(
                name="ObjectSampler"
            )
            for nut_name, default_y_range in zip(
                nut_names, ([-0.11, -0.09],)
            ):  # Only one range for the round nut
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{nut_name}Sampler",
                        x_range=[0.07, 0.09],
                        y_range=default_y_range,
                        rotation=[3.0, 3.28],
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )
                )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        # Only create the round nut
        nut = RoundNutObject(name="RoundNut")
        self.nuts.append(nut)

        # Add this nut to the placement initializer
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            # Add the round nut to the sampler
            self.placement_initializer.add_objects_to_sampler(
                sampler_name="RoundNutSampler", mujoco_objects=nut
            )
        else:
            # This is assumed to be a flat sampler, so we just add the nut to this sampler
            self.placement_initializer.add_objects(nut)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # Call the parent reset_internal from ManipulationEnv
        ManipulationEnv._reset_internal(self)

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

        # Set the obj_to_use to be our single round nut
        self.obj_to_use = "RoundNut"

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.nuts))

        # Make sure to update sensors' active and enabled states
        for i, sensor_names in self.nut_id_to_sensors.items():
            for name in sensor_names:
                # Set all of these sensors to be enabled and active if this is the active nut, else False
                self._observables[name].set_enabled(i == self.nut_id)
                self._observables[name].set_active(i == self.nut_id)

    def reward(self, action=None):
        """
        The reward function of the environment.

        Args:
            action (np.ndarray, optional): The action taken by the agent. Defaults to None.

        Returns:
            int: The reward
        """
        return 0

    # Developer may want to override this in their child class
    def _check_success(self):
        """
        Always returns False to prevent the environment from terminating.
        """
        return False

    def get_available_body_names(self):
        return self.sim.model.body_names

    def get_nut_pos(self, name: str = "RoundNut_main"):
        nut_id = self.sim.model.body_name2id(name)
        nut_pos = self.sim.data.body_xpos[nut_id]
        return nut_pos

    def get_nut_ori(self, name: str = "RoundNut_main"):
        nut_id = self.sim.model.body_name2id(name)
        nut_ori = self.sim.data.body_xquat[nut_id]
        return nut_ori

    def get_gripper_tip_pos(self, name: str = "gripper0_right_finger_joint1_tip"):
        tip_id = self.sim.model.body_name2id(name)
        tip_pos = self.sim.data.body_xpos[tip_id]
        return tip_pos

    def get_object_position(self, target, target_type):
        if isinstance(target, MujocoModel):
            return self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            return self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            return self.sim.data.get_site_xpos(target)
        else:
            return self.sim.data.get_geom_xpos(target)

    def get_distance_from_gripper_to_nut_handle(self, gripper=None, nut=None) -> float:
        if gripper is None:
            gripper = self.robots[0].gripper
        if nut is None:
            nut = self.nuts[0]
        return self._gripper_to_target(
            gripper=gripper,
            target=nut.important_sites["handle"],
            target_type="site",
            return_distance=True,
        )

    def check_grasp(self, gripper=None, nut=None):
        if gripper is None:
            gripper = self.robots[0].gripper
        if nut is None:
            nut = self.nuts[0]
        return self._check_grasp(
            gripper=gripper,
            object_geoms=nut.contact_geoms,
        )