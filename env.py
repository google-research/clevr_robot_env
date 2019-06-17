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

"""The CLEVR-ROBOT environment."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from gym import spaces
from gym import utils
import numpy as np

from third_party.clevr_robot_env_utils.generate_question import generate_question_from_scene_struct
import third_party.clevr_robot_env_utils.generate_scene as gs
import third_party.clevr_robot_env_utils.question_engine as qeng

import clevr_robot_env.utils.load_utils as load_utils
from clevr_robot_env.utils.xml_utils import convert_scene_to_xml

try:
  import cv2
  import mujoco
  from gym.envs.mujoco import mujoco_env
except ImportError as e:
  print(e)

DEFAULT_XML_PATH = os.path.join(__file__, 'assets/clevr_default.xml')
FIXED_PATH = os.path.join(__file__, 'templates', '10_fixed_objective.pkl')

# metadata
DEFAULT_METADATA_PATH = os.path.join(__file__, 'metadata', 'metadata.json')
VARIABLE_OBJ_METADATA_PATH = os.path.join(__file__, 'metadata',
                                          'variable_obj_meta_data.json')

# template_path
EVEN_Q_DIST_TEMPLATE = os.path.join(
    __file__, 'templates/even_question_distribution.json')
VARIABLE_OBJ_TEMPLATE = os.path.join(__file__, 'templates',
                                     'variable_object.json')


# fixed discrete action set
DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1], [0.8, 0.8], [-0.8, 0.8],
              [0.8, -0.8], [-0.8, -0.8]]
X_RANGE, Y_RANGE = 0.7, 0.35


def _create_discrete_action_set():
  discrete_action_set = []
  for d in DIRECTIONS:
    for x in [-X_RANGE + i * X_RANGE / 5. for i in range(10)]:
      for y in [-Y_RANGE + i * 0.12 for i in range(10)]:
        discrete_action_set.append([[x, y], d])
  return discrete_action_set


DISCRETE_ACTION_SET = _create_discrete_action_set()

# cardinal vectors
# TODO: ideally this should be packaged into scene struct
four_cardinal_vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
four_cardinal_vectors = np.array(four_cardinal_vectors, dtype=np.float32)
four_cardinal_vectors_names = ['front', 'behind', 'left', 'right']


class ClevrEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  """ClevrEnv."""

  def __init__(self,
               maximum_episode_steps=100,
               xml_path=None,
               metadata_path=None,
               template_path=None,
               num_object=5,
               agent_type='pm',
               random_start=False,
               fixed_objective=True,
               description_num=15,
               action_type='continuous',
               obs_type='direct',
               use_movement_bonus=False,
               direct_obs=False,
               reward_scale=1.0,
               frame_skip=20,
               shape_val=0.25,
               min_move_dist=0.05,
               resolution=64,
               use_synonyms=False,
               min_change_th=0.26,
               use_polar=False,
               use_subset_instruction=False,
               systematic_generalization=False,
               suppress_other_movement=False,
               top_down_view=False,
               variable_scene_content=False):

    utils.EzPickle.__init__(self)
    initial_xml_path = DEFAULT_XML_PATH
    self.obj_name = []
    self.action_type = action_type
    self.use_movement_bonus = use_movement_bonus
    self.direct_obs = direct_obs
    self.obs_type = obs_type
    self.num_object = num_object
    self.variable_scene_content = variable_scene_content
    self.cache_valid_questions = variable_scene_content
    self.checker_board = variable_scene_content
    self.reward_scale = reward_scale
    self.shape_val = shape_val
    self.min_move_dist = min_move_dist
    self.res = resolution
    self.use_synonyms = use_synonyms
    self.min_change_th = min_change_th
    self.use_polar = use_polar
    self.suppress_other_movement = suppress_other_movement

    if use_subset_instruction and systematic_generalization:
      train, test = load_utils.create_systematic_generalization_split()
    elif use_subset_instruction and not systematic_generalization:
      train, test = load_utils.create_train_test_question_split()
    else:
      train, test = load_utils.load_all_question(), None

    self.all_questions, self.held_out_questions = train, test
    self.all_question_num = len(self.all_questions)

    # loading meta data
    if metadata_path is None:
      metadata_path = DEFAULT_METADATA_PATH

    if self.variable_scene_content:
      print('loading variable input metadata')
      metadata_path = VARIABLE_OBJ_METADATA_PATH

    with open(metadata_path, 'r') as metadata_file:
      self.clevr_metadata = json.load(metadata_file)

    functions_by_name = {}
    for func in self.clevr_metadata['functions']:
      functions_by_name[func['name']] = func
    self.clevr_metadata['_functions_by_name'] = functions_by_name

    # information regarding question template
    if template_path is None:
      template_path = EVEN_Q_DIST_TEMPLATE
    if self.variable_scene_content:
      print('loading variable input template')
      template_path = VARIABLE_OBJ_TEMPLATE

    self.template_num = 0
    self.templates = {}
    fn = 'general_template'
    with open(template_path, 'r') as template_file:
      for i, template in enumerate(json.load(template_file)):
        self.template_num += 1
        key = (fn, i)
        self.templates[key] = template
    print('Read {} templates from disk'.format(self.template_num))

    # setting up camera transformation
    self.w2c, self.c2w = gs.camera_transformation_from_pose(90, -45)

    # sample a random scene and struct
    self.scene_graph, self.scene_struct = self.sample_random_scene()

    # total number of colors and shapes
    def one_hot_encoding(key_to_idx, max_length):
      encoding_map = {}
      for k in key_to_idx:
        one_hot_vector = [0] * max_length
        one_hot_vector[key_to_idx[k]] = 1
        encoding_map[k] = one_hot_vector
      return encoding_map

    mdata_types = self.clevr_metadata['types']
    self.color_n = len(mdata_types['Color'])
    self.color_to_idx = {c: i for i, c in enumerate(mdata_types['Color'])}
    self.color_to_one_hot = one_hot_encoding(self.color_to_idx, self.color_n)
    self.shape_n = len(mdata_types['Shape'])
    self.shape_to_idx = {s: i for i, s in enumerate(mdata_types['Shape'])}
    self.shape_to_one_hot = one_hot_encoding(self.shape_to_idx, self.shape_n)
    self.size_n = len(mdata_types['Size'])
    self.size_to_idx = {s: i for i, s in enumerate(mdata_types['Size'])}
    self.size_to_one_hot = one_hot_encoding(self.size_to_idx, self.size_n)
    self.mat_n = len(mdata_types['Material'])
    self.mat_to_idx = {s: i for i, s in enumerate(mdata_types['Material'])}
    self.mat_to_one_hot = one_hot_encoding(self.mat_to_idx, self.mat_n)

    # generate initial set of description from the scene graph
    self.description_num = description_num
    self.descriptions, self.full_descriptions = None, None
    self._update_description()
    self.obj_description = []
    self._update_object_description()

    mujoco_env.MujocoEnv.__init__(
        self,
        initial_xml_path,
        frame_skip,
        max_episode_steps=maximum_episode_steps,
        reward_threshold=0.,
    )

    # name of geometries in the scene
    self.obj_name = ['obj{}'.format(i) for i in range(self.num_object)]

    self.discrete_action_set = DISCRETE_ACTION_SET
    self.perfect_action_set = []
    for i in range(self.num_object):
      for d in DIRECTIONS:
        self.perfect_action_set.append(np.array([i] + d))

    # set discrete action space
    if self.action_type == 'discrete':
      self._action_set = DISCRETE_ACTION_SET
      self.action_space = spaces.Discrete(len(self._action_set))
    elif self.action_type == 'perfect':
      self._action_set = self.perfect_action_set
      self.action_space = spaces.Discrete(len(self._action_set))
    elif self.action_type == 'continuous':
      self.action_space = spaces.Box(
          low=-1.0, high=1.1, shape=[4], dtype=np.float32)
    else:
      raise ValueError('{} is not a valid action type'.format(action_type))

    # setup camera and observation space
    self.camera = mujoco.MovableCamera(self.physics, height=300, width=300)
    self._top_down_view = top_down_view
    if top_down_view:
      camera_pose = self.camera.get_pose()
      self.camera.set_pose(camera_pose.lookat, camera_pose.distance,
                           camera_pose.azimuth, -90)
    self.camera_setup()

    if self.direct_obs:
      self.observation_space = spaces.Box(
          low=np.concatenate(zip([-0.6] * num_object, [-0.4] * num_object)),
          high=np.concatenate(zip([0.6] * num_object, [0.6] * num_object)),
          dtype=np.float32)
    else:
      self.observation_space = spaces.Box(
          low=0, high=255, shape=(self.res, self.res, 3), dtype=np.uint8)

    # agent type and randomness of starting location
    self.agent_type = agent_type
    self.random_start = random_start

    if not self.random_start:
      curr_scene_xml = convert_scene_to_xml(
          self.scene_graph,
          agent=self.agent_type,
          checker_board=self.checker_board)
    else:
      random_loc = '{} {} -0.2'.format(
          random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
      curr_scene_xml = convert_scene_to_xml(
          self.scene_graph,
          agent=self.agent_type,
          agent_start_loc=random_loc,
          checker_board=self.checker_board)
    self.load_xml_string(curr_scene_xml)

    self.valid_questions = []

    # counter for reset
    self.reset(True)
    self.curr_step = 0
    self.current_goal_text, self.current_goal = self.sample_goal()
    self.achieved_last_step = []
    self.achieved_last_step_program = []
    print('CLEVR-ROBOT environment initialized.')

  def step(self,
           a,
           record_achieved_goal=False,
           goal=None,
           atomic_goal=False,
           update_des=False):
    """Take step a in the environment."""

    info = {}

    if not self.obj_name:
      self.do_simulation([0, 0], self.frame_skip)
      return self.get_obs(), 0, False, None

    # record questions that are currently false for relabeling
    currently_false = []
    if record_achieved_goal:
      if not self.cache_valid_questions:
        candidates = self.all_questions
      else:
        candidates = self.valid_questions
      random.shuffle(candidates)
      false_question_count = 0

      for q, p in candidates:
        if false_question_count > 128 and self.cache_valid_questions:
          break
        full_answer = self.answer_question(p, True)
        fixed_object_idx, fixed_object_loc = self._get_fixed_object(full_answer)
        if not full_answer[-1] and fixed_object_loc is not None:
          currently_false.append((q, p, fixed_object_idx, fixed_object_loc))
          false_question_count += 1

      random.shuffle(currently_false)

    if goal:
      full_answer = self.answer_question(goal, True)
      g_obj_idx, g_obj_loc = self._get_fixed_object(full_answer)

    curr_state = np.array([self.get_body_com(name) for name in self.obj_name])

    if self.action_type == 'discrete':
      self.step_discrete(a)
    elif self.action_type == 'perfect' and self.obs_type != 'order_invariant':
      self.step_perfect_noi(a)
    elif self.action_type == 'perfect' and self.obs_type == 'order_invariant':
      self.step_perfect_oi(a)
    elif self.action_type == 'continuous':
      self.step_continuous(a)

    new_state = np.array([self.get_body_com(name) for name in self.obj_name])
    displacement_vector = np.stack(
        [a - b for a, b in zip(curr_state, new_state)])
    atomic_movement_description = self._get_atomic_object_movements(
        displacement_vector)

    self.curr_step += 1
    self._update_scene()
    if update_des:
      self._update_description()
      info['descriptions'] = self.descriptions
      info['full_descriptions'] = self.full_descriptions

    if record_achieved_goal:
      self.achieved_last_step = []
      self.achieved_last_step_program = []
      for q, p, obj_idx, obj_loc in currently_false:
        # fixed_object_idx
        obj_cur_loc = np.array(self.scene_graph[obj_idx]['3d_coords'])[:-1]
        # checking the first object has not been moved
        dispalcement = np.linalg.norm(obj_cur_loc - obj_loc)
        if self.answer_question(p) and dispalcement < self.min_change_th:
          self.achieved_last_step.append(q)
          self.achieved_last_step_program.append(p)

    if record_achieved_goal and atomic_goal:
      self.achieved_last_step += atomic_movement_description

    if not goal:
      r = self._reward()
    elif not self.suppress_other_movement:
      g_obj_cur_loc = np.array(self.scene_graph[g_obj_idx]['3d_coords'])[:-1]
      dispalcement = np.linalg.norm(g_obj_cur_loc - g_obj_loc)
      r = self.answer_question(goal)
      r = r and dispalcement < (self.min_change_th + 0.1)
      r = float(r)
      if self.use_movement_bonus and atomic_movement_description and r < 1.0:
        r += self.shape_val
    else:
      r = float(self.answer_question(goal))
      if self.use_movement_bonus and atomic_movement_description and r < 1.0:
        r += self.shape_val
      if r >= 1.0:
        r += self._get_obj_movement_bonus(g_obj_idx, displacement_vector)

    done = self.curr_step >= self.max_episode_steps

    obs = self.get_obs()

    return obs, r, done, info

  def teleport(self, loc):
    """Teleport the agent to loc."""
    # Location might be 2D because of no vertical movement
    curr_loc = self.get_body_com('point_mass')[:len(loc)]
    dsp_vec = loc - curr_loc
    qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
    qpos[-2:] = qpos[-2:] + dsp_vec
    qvel[-2:] = np.zeros(2)
    self.set_state(qpos, qvel)

  def step_discrete(self, a):
    """Take discrete step by teleporting and then push."""
    a = int(a)
    action = self.discrete_action_set[a]
    new_loc = np.array(action[0])
    self.teleport(new_loc)
    self.do_simulation(np.array(action[1]) * 1.1, int(self.frame_skip * 2.0))

  def step_perfect_noi(self, a):
    """Take a perfect step by teleporting and then push in fixed obj setting."""
    a = int(a)
    action = self._action_set[a]
    obj = action[0]
    obj_loc = self.get_body_com(self.obj_name[int(obj)])
    push_start = np.array(obj_loc)[:-1] - 0.15 * action[1:]
    dsp_vec = push_start - self.get_body_com('point_mass')[:-1]
    qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
    qpos[-2:] = qpos[-2:] + dsp_vec
    qvel[-2:] = np.zeros(2)
    self.set_state(qpos, qvel)
    self.do_simulation(action[1:] * 1.0, int(self.frame_skip * 2.0))

  def step_perfect_oi(self, a):
    """Take a perfect step by teleporting and then push in fixed obj setting."""
    obj_selection, dir_selection = int(a[0]), int(a[1])
    direction = np.array(DIRECTIONS[dir_selection])
    obj_loc = self.scene_graph[obj_selection]['3d_coords'][:-1]
    push_start = np.array(obj_loc) - 0.15 * direction
    dsp_vec = push_start - self.get_body_com('point_mass')[:-1]
    qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
    qpos[-2:] = qpos[-2:] + dsp_vec
    qvel[-2:] = np.zeros(2)
    self.set_state(qpos, qvel)
    self.do_simulation(direction * 1.0, int(self.frame_skip * 2.0))

  def step_continuous(self, a):
    """Take a continuous version of step discrete."""
    a = np.squeeze(a)
    x, y, theta, r = a[0] * 0.7, a[1] * 0.7, a[2] * np.pi, a[3]
    direction = np.array([np.cos(theta), np.sin(theta)]) * 1.2
    duration = int((r + 1.0) * self.frame_skip * 3.0)
    new_loc = np.array([x, y])
    qpos, qvel = self.physics.data.qpos, self.physics.data.qvel
    qpos[-2:], qvel[-2:] = new_loc, np.zeros(2)
    self.set_state(qpos, qvel)
    curr_loc = self.get_body_com('point_mass')
    dist = [curr_loc - self.get_body_com(name) for name in self.obj_name]
    dist = np.min(np.linalg.norm(dist, axis=1))
    self.do_simulation(direction, duration)

  def reset(self, new_scene_content=True):
    """Reset with a random configuration."""
    if new_scene_content or not self.variable_scene_content:
      # sample a random scene and struct
      self.scene_graph, self.scene_struct = self.sample_random_scene()
    else:
      # randomly perturb existing objects in the scene
      new_graph = gs.randomly_perturb_objects(self.scene_struct,
                                              self.scene_graph)
      self.scene_graph = new_graph
      self.scene_struct['objects'] = self.scene_graph
      self.scene_struct['relationships'] = gs.compute_relationship(
          self.scene_struct)

    # Generate initial set of description from the scene graph.
    self.descriptions, self.full_descriptions = None, None
    self._update_description()
    self.curr_step = 0

    if not self.random_start:
      curr_scene_xml = convert_scene_to_xml(
          self.scene_graph,
          agent=self.agent_type,
          checker_board=self.checker_board)
    else:
      random_loc = '{} {} -0.2'.format(
          random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
      curr_scene_xml = convert_scene_to_xml(
          self.scene_graph,
          agent=self.agent_type,
          agent_start_loc=random_loc,
          checker_board=self.checker_board)
    self.load_xml_string(curr_scene_xml)

    if self.variable_scene_content and self.cache_valid_questions and new_scene_content:
      self.valid_questions = self.sample_valid_questions(100)
      if len(self.valid_questions) < 5:
        print('rerunning reset because valid question count is small')
        return self.reset(True)
      self.current_goal_text, self.current_goal = self.sample_goal()

    self._update_object_description()

    return self.get_obs()

  def get_obs(self):
    """Returns the state representation of the current scene."""
    if self.direct_obs and self.obs_type != 'order_invariant':
      return self.get_direct_obs()
    elif self.direct_obs and self.obs_type == 'order_invariant':
      return self.get_order_invariant_obs()
    else:
      return self.get_image_obs()

  def get_direct_obs(self):
    """Returns the direct state observation."""
    all_pos = np.array([self.get_body_com(name) for name in self.obj_name])
    has_obj = len(all_pos.shape) > 1
    all_pos = all_pos[:, :-1] if has_obj else np.zeros(2 * self.num_object)
    return all_pos.flatten()

  def get_image_obs(self):
    """Returns the image observation."""
    frame = self.render(mode='rgb_array')
    frame = cv2.resize(
        frame, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC)
    return frame / 255.

  def get_order_invariant_obs(self):
    """Returns the order invariant observation.

    The returned vector will be a 2D array where the first axis is the object
    in the scene (which can be varying) and the second axis is the object
    description. Each object's description contains its x-y location and
    one-hot representation of its attributes (color, shape etc).
    """
    obs = []
    for obj in self.scene_graph:
      obj_vec = list(obj['3d_coords'][:-1])
      obj_vec += self.size_to_one_hot[obj['size']]
      obj_vec += self.color_to_one_hot[obj['color']]
      obj_vec += self.mat_to_one_hot[obj['material']]
      obj_vec += self.shape_to_one_hot[obj['shape']]
      obs.append(obj_vec)
    return np.array(obs)

  def get_achieved_goals(self):
    """Get goal that are achieved from the latest interaction."""
    return self.achieved_last_step

  def get_achieved_goal_programs(self):
    """Get goal programs that are achieved from the latest interaction."""
    return self.achieved_last_step_program

  def set_goal(self, goal_text, goal_program):
    """Set the goal to be used in standard RL settings."""
    self.current_goal_text = goal_text
    self.current_goal = goal_program

  def sample_random_scene(self):
    """Sample a random scene base on current viewing angle."""
    if self.variable_scene_content:
      return gs.generate_scene_struct(self.c2w, self.num_object,
                                      self.clevr_metadata)
    else:
      return gs.generate_scene_struct(self.c2w, self.num_object)

  def sample_goal(self):
    """Sample a currently false statement and its corresponding text."""
    candidate_objective = self.all_questions
    if self.cache_valid_questions:
      candidate_objective = self.valid_questions
    random.shuffle(candidate_objective)
    for g, gp in candidate_objective:
      if not self.answer_question(gp):
        self.all_goals_satisfied = False
        return g, gp
    print('All goal are satisfied.')
    goal, goal_program = random.choice(candidate_objective)
    self.all_goals_satisfied = True
    return goal, goal_program

  def sample_random_action(self):
    """Sample a random action for the environment."""
    if self.obs_type == 'order_invariant' and self.action_type == 'perfect':
      action = [
          np.random.randint(low=0, high=self.num_object),
          np.random.randint(low=0, high=len(DIRECTIONS))
      ]
      return np.array(action)
    else:
      return self.action_space.sample()

  def sample_valid_questions(self, iterations=50):
    """Sample valid questions for the current scene content."""
    current_graph = self.scene_graph
    all_q = []
    for _ in range(iterations):
      new_graph = gs.randomly_perturb_objects(self.scene_struct, current_graph)
      self.scene_struct['objects'] = new_graph
      self.scene_struct['relationships'] = gs.compute_relationship(
          self.scene_struct)
      self._update_description()
      all_q += self.full_descriptions
    for q in all_q:
      for node in q['program']:
        if '_output' in node:
          del node['_output']
    # get question that are unique and can be satisfied
    unique_and_feasible = {}
    for q in all_q:
      q_is_unique = repr(q['program']) not in unique_and_feasible
      if q['answer'] is True and q_is_unique:
        unique_and_feasible[repr(q['program'])] = q
    valid_q = []
    for q in unique_and_feasible:
      valid_q.append((unique_and_feasible[q]['question'],
                      unique_and_feasible[q]['program']))
    self.scene_struct['objects'] = current_graph
    self.scene_struct['relationships'] = gs.compute_relationship(
        self.scene_struct)
    return valid_q

  def answer_question(self, program, all_outputs=False):
    """Answer a functional program on the current scene."""
    return qeng.answer_question({'nodes': program},
                                self.clevr_metadata,
                                self.scene_struct,
                                cache_outputs=False,
                                all_outputs=all_outputs)

  def convert_order_invariant_to_direct(self, order_invariant_obs):
    """Converts the order invariant observation to state observation."""
    return order_invariant_obs[:, :2].flatten()

  def load_xml_string(self, xml_string):
    """Load the model into physics specified by a xml string."""
    self.physics.reload_from_xml_string(xml_string)

  def load_xml_path(self, xml_path):
    """Load the model into physics specified by a xml path."""
    self.physics.reload_from_xml_path(xml_path)

  def get_description(self):
    """Update and return the current scene description."""
    self._update_description()
    return self.descriptions, self.full_descriptions

  def _update_description(self, custom_n=None):
    """Update the text description of the current scene."""
    gq = generate_question_from_scene_struct
    dn = self.description_num if not custom_n else custom_n
    tn = self.template_num
    self.descriptions, self.full_descriptions = gq(
        self.scene_struct,
        self.clevr_metadata,
        self.templates,
        templates_per_image=tn,
        instances_per_template=dn,
        use_synonyms=self.use_synonyms)

  def _update_scene(self):
    """Update the scene description of the current scene."""
    self.previous_scene_graph = self.scene_graph
    for i, name in enumerate(self.obj_name):
      self.scene_graph[i]['3d_coords'] = tuple(self.get_body_com(name))
    self.scene_struct['objects'] = self.scene_graph
    self.scene_struct['relationships'] = gs.compute_relationship(
        self.scene_struct, use_polar=self.use_polar)

  def _update_object_description(self):
    """Update the scene description of the current scene."""
    self.obj_description = []
    for i in range(len(self.obj_name)):
      obj = self.scene_graph[i]
      color = obj['color']
      shape = obj['shape_name']
      material = obj['material']
      self.obj_description.append(' '.join([color, material, shape]))

  def _get_atomic_object_movements(self, displacement):
    """Get a list of sentences that describe the movements of object."""
    atomic_sentence = []
    for o, d in zip(self.obj_description, displacement):
      # TODO: this might need to be removed for stacking
      d_norm = np.linalg.norm(d[:-1])  # not counting height in displacement
      if d_norm > self.min_move_dist:
        max_d = np.argmax(np.dot(four_cardinal_vectors, d))
        atomic_sentence.append(' '.join(
            [o, 'to', four_cardinal_vectors_names[max_d]]))
    return atomic_sentence

  def _get_fixed_object(self, answer):
    """Get the index and location of object that should be fixed in a query."""
    index, loc = -1, None
    for i, a in enumerate(answer):
      if a is True:
        index = random.choice(answer[i - 1])
      elif isinstance(a, float) or isinstance(a, int):
        index = answer[i]
        break
    if index >= 0:
      loc = np.array(self.scene_graph[index]['3d_coords'])[:-1]
    return index, loc

  def _get_obj_movement_bonus(self, fixed_obj_idx, displacement_vector):
    """Get the bonus reward for not moving other object."""
    del fixed_obj_idx
    norm = np.linalg.norm(displacement_vector, axis=-1)
    total_norm = norm.sum()
    return 0.5 * np.exp(-total_norm * 7)

  def _reward(self):
    return float(self.answer_question(self.current_goal))
