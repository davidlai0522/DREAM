from pddl_rl_robot.simulation.peg_and_hole_base_env import NutAssembly, PegsArena
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.wrappers import GymWrapper
from robosuite.models.objects import RoundNutObject
import numpy as np

class TwoPegOneRoundNut(NutAssembly):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="round", **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        ManipulationEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
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
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for nut_name, default_y_range in zip(nut_names, ([-0.11, -0.09],)):  # Only one range for the round nut
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
            self.placement_initializer.add_objects_to_sampler(sampler_name="RoundNutSampler", mujoco_objects=nut)
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
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
        return 0

    def _check_success(self):
        """
        Always returns False to prevent the environment from terminating.
        """
        return False
