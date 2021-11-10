# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper that makes dm_control look like gym mujoco."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dm_control import mujoco
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MujocoEnv(gym.Env):
  """Custom Mujoco environment that uses dm control's wrapper."""

  def __init__(self, model_path, frame_skip, max_episode_steps=None,
               reward_threshold=None):

    if model_path.startswith('/'):
      fullpath = model_path
    else:
      fullpath = os.path.join(
          os.path.abspath(os.path.dirname(__file__)), 'assets', model_path)

    if not os.path.exists(fullpath):
      raise IOError('File %s does not exist' % fullpath)

    self.frame_skip = frame_skip
    self.physics = mujoco.Physics.from_xml_path(fullpath)
    self.camera = mujoco.MovableCamera(self.physics, height=480, width=640)

    self.viewer = None

    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': int(np.round(1.0 / self.dt))
    }

    self.init_qpos = self.physics.data.qpos.ravel().copy()
    self.init_qvel = self.physics.data.qvel.ravel().copy()
    observation, _, done, _ = self.step(np.zeros(self.physics.model.nu))
    assert not done
    self.obs_dim = observation.size

    bounds = self.physics.model.actuator_ctrlrange.copy()
    low = bounds[:, 0]
    high = bounds[:, 1]
    self.action_space = spaces.Box(low, high, dtype=np.float32)

    high = np.inf * np.ones(self.obs_dim)
    low = -high
    self.observation_space = spaces.Box(low, high, dtype=np.float32)

    self.max_episode_steps = max_episode_steps
    self.reward_threshold = reward_threshold

    self.seed()
    self.camera_setup()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset_model(self):
    """Reset the robot degrees of freedom (qpos and qvel)."""
    raise NotImplementedError('Implement this in each subclass.')

  def viewer_setup(self):
    """This method is called when the viewer is initialized and after all reset.

    Optionally implement this method, if you need to tinker with camera position
    and so forth.
    """
    pass

  def set_state(self, qpos, qvel):
    assert qpos.shape == (self.physics.model.nq,) and qvel.shape == (
        self.physics.model.nv,)
    assert self.physics.get_state().size == qpos.size + qvel.size
    state = np.concatenate([qpos, qvel], 0)
    with self.physics.reset_context():
      self.physics.set_state(state)

  @property
  def dt(self):
    return self.physics.model.opt.timestep * self.frame_skip

  def do_simulation(self, ctrl, n_frames):
    self.physics.set_control(ctrl)
    for _ in range(n_frames):
      self.physics.step()

  def render(self, mode='human'):
    if mode == 'rgb_array':
      data = self.camera.render()
      return np.copy(data)  # render reuses the same memory space.
    elif mode == 'human':
      raise NotImplementedError('Interactive rendering not implemented yet.')

  def get_body_com(self, body_name):
    idx = self.physics.model.name2id(body_name, 1)
    return self.physics.data.subtree_com[idx]

  def state_vector(self):
    return np.concatenate(
        [self.physics.data.qpos.flat, self.physics.data.qvel.flat])

  def get_state(self):
    return np.array(self.physics.data.qpos.flat), np.array(
        self.physics.data.qvel.flat)

  def camera_setup(self):
    pass  # override this to set up camera
