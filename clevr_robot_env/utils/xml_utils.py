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

"""XML utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree as ET

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

red_metal_path = os.path.join(parent_dir, 'assets/textures/metal_red.png')
cyan_metal_path = os.path.join(parent_dir, 'assets/textures/metal_cyan.png')
purple_metal_path = os.path.join(parent_dir, 'assets/textures/metal_purple.png')
green_metal_path = os.path.join(parent_dir, 'assets/textures/metal_green.png')
blue_metal_path = os.path.join(parent_dir, 'assets/textures/metal_blue.png')


texture = {
    'blue': [('name', 'tex_blue'), ('type', '2d'), ('file', blue_metal_path)],
    'red': [('name', 'tex_red'), ('type', '2d'), ('file', red_metal_path)],
    'cyan': [('name', 'tex_cyan'), ('type', '2d'), ('file', cyan_metal_path)],
    'green': [('name', 'tex_green'), ('type', '2d'),
              ('file', green_metal_path)],
    'purple': [('name', 'tex_purple'), ('type', '2d'),
               ('file', purple_metal_path)],
    'checker': [('builtin', 'checker'), ('height', '20'), ('name', 'texplane'),
                ('rgb1', '0.4 0.4 0.4'), ('rgb2', '0.8 0.8 0.8'),
                ('type', '2d'), ('width', '20')]
}

custom_material = {
    'blue_metal': [('name', 'blue_metal'), ('specular', '1'),
                   ('shininess', '1'), ('texture', 'tex_blue'),
                   ('emission', '0.4')],
    'red_metal': [('name', 'red_metal'), ('specular', '1'), ('shininess', '1'),
                  ('texture', 'tex_red'), ('emission', '0.4')],
    'cyan_metal': [('name', 'cyan_metal'), ('specular', '1'),
                   ('shininess', '1'), ('texture', 'tex_cyan'),
                   ('emission', '0.4')],
    'green_metal': [('name', 'green_metal'), ('specular', '1'),
                    ('shininess', '1'), ('texture', 'tex_green'),
                    ('emission', '0.4')],
    'purple_metal': [('name', 'purple_metal'), ('specular', '1'),
                     ('shininess', '1'), ('texture', 'tex_purple'),
                     ('emission', '0.4')],
    'matplane': [('name', 'matplane'), ('reflectance', '0.05'),
                 ('shininess', '0.5'), ('specular', '0.25'),
                 ('texrepeat', '5 5'), ('texture', 'texplane')]
}


def convert_scene_to_xml(scene,
                         model_name='scene',
                         agent='pm',
                         agent_start_loc='0 0 -0.2',
                         checker_board=False):
  """Convert a scene to a xml string."""
  # root
  root = ET.Element('mujoco')
  root.set('model', model_name)
  # compiler
  compiler_attribute = [('inertiafromgeom', 'true'), ('angle', 'radian'),
                        ('coordinate', 'local')]
  compiler = ET.SubElement(root, 'compiler')
  set_attribute(compiler, compiler_attribute)
  # option
  option_attribute = [('timestep', '0.01'), ('gravity', '0 0 -9.81'),
                      ('iterations', '20'), ('integrator', 'Euler')]
  option = ET.SubElement(root, 'option')
  set_attribute(option, option_attribute)
  # default
  default = ET.SubElement(root, 'default')
  default_joint = ET.SubElement(default, 'joint')
  default_geom = ET.SubElement(default, 'geom')
  default_joint_attribute = [('armature', '0.04'), ('damping', '1'),
                             ('limited', 'true')]
  set_attribute(default_joint, default_joint_attribute)
  default_geom_attribute = [('friction', '.8 .1 .1'), ('density', '300'),
                            ('margin', '0.002'), ('condim', '3'),
                            ('contype', '0'), ('conaffinity', '0')]
  set_attribute(default_geom, default_geom_attribute)
  # asset
  asset = ET.SubElement(root, 'asset')
  material = {
      'rubber': [('name', 'rubber'), ('specular', '0.0'), ('shininess', '0.1'),
                 ('reflectance', '0.0')],
      'metal': [('name', 'metal'), ('specular', '1.0'), ('shininess', '1.0'),
                ('reflectance', '1.0'), ('emission', '0.2')]
  }
  for m in material:
    m_node = ET.SubElement(asset, 'material')
    set_attribute(m_node, material[m])

  for m in texture:
    m_node = ET.SubElement(asset, 'texture')
    set_attribute(m_node, texture[m])

  for m in custom_material:
    m_node = ET.SubElement(asset, 'material')
    set_attribute(m_node, custom_material[m])

  # world body
  world_body = ET.SubElement(root, 'worldbody')
  light = ET.SubElement(world_body, 'light')
  set_attribute(light, [('diffuse', '.5 .5 .5'), ('pos', '0 0 3'),
                        ('dir', '0 0 -1')])
  # table
  table = ET.SubElement(world_body, 'geom')
  table_attr = [('name', 'table'), ('type', 'plane'), ('pos', '0 0.5 -0.325'),
                ('size', '1 1 0.1'), ('contype', '1'), ('conaffinity', '1')]
  if checker_board:
    table_attr += [('material', 'matplane')]
  set_attribute(table, table_attr)

  # left invisible plane
  plane_left = ET.SubElement(world_body, 'geom')
  set_attribute(plane_left, [('name', 'left_plane'), ('type', 'plane'),
                             ('pos', '-0.6 0.0 0.0'), ('size', '2 2 0.1'),
                             ('contype', '1'), ('conaffinity', '0'),
                             ('rgba', '1.0 0.5 1.0 0.0'),
                             ('euler', '0. 1.57 0.')])
  # right invisible plane
  plane_right = ET.SubElement(world_body, 'geom')
  set_attribute(plane_right, [('name', 'right_plane'), ('type', 'plane'),
                              ('pos', '0.6 0.0 0.0'), ('size', '2 2 0.1'),
                              ('contype', '1'), ('conaffinity', '0'),
                              ('rgba', '0.5 1.0 1.0 0.0'),
                              ('euler', '0. -1.57 0.')])

  # right invisible plane
  plane_front = ET.SubElement(world_body, 'geom')
  set_attribute(plane_front, [('name', 'front_plane'), ('type', 'plane'),
                              ('pos', '0.0 0.7 0.0'), ('size', '2 2 0.1'),
                              ('contype', '1'), ('conaffinity', '0'),
                              ('rgba', '1.0 1.0 0.5 0.0'),
                              ('euler', '1.57 0. 0.')])

  # right invisible plane
  plane_behind = ET.SubElement(world_body, 'geom')
  set_attribute(plane_behind, [('name', 'behind_plane'), ('type', 'plane'),
                               ('pos', '0.0 -0.3 0.0'), ('size', '2 2 0.1'),
                               ('contype', '1'), ('conaffinity', '0'),
                               ('rgba', '1.0 1.0 0.5 0.0'),
                               ('euler', '-1.57 0. 0.')])

  set_scene_object(world_body, scene)
  # gripper and actuator
  if agent == 'pm':
    set_point_mass(root, world_body, agent_start_loc)
  elif agent == 'simple_gripper':
    set_simple_gripper(root, world_body)

  return ET.tostring(root)


def set_attribute(node, attribute_pairs):
  for k, v in attribute_pairs:
    node.set(k, v)


def set_scene_object(worldbody, scene):
  """Set the xml element of a scene configuration."""
  count = 0
  friction_joint_1 = [('name', 'ph1'), ('type', 'free'), ('pos', '0 0 0'),
                      ('damping', '0.75'), ('limited', 'false')]
  geom_attr = [('rgba', '1 1 1 1'), ('type', 'cylinder'),
               ('size', '0.05 0.05 0.05'), ('density', '2'), ('contype', '1'),
               ('conaffinity', '1'), ('material', 'rubber')]
  for body in scene:
    loc = body['3d_coords']
    shape = body['shape_name']
    body_node = ET.SubElement(worldbody, 'body')
    body_node.set('name', 'obj{}'.format(count))
    loc_str = ' '.join([str(loc[0]), str(loc[1]), str(-0.325 + loc[2])])
    body_node.set('pos', loc_str)
    # body geometry
    if shape == 'cylinder':
      geom_attr[2] = ('size',
                      '{} {} 0.05'.format(str(loc[2]), str(loc[2] / 1.2)))
    elif shape == 'sphere':
      geom_attr[2] = ('size', '{} 0.05 0.05'.format(str(loc[2])))
    elif shape == 'box':
      geom_attr[2] = ('size',
                      '{} {} {}'.format(str(loc[2]), str(loc[2]), str(loc[2])))
    if body['size'] == 'large':
      geom_attr[3] = ('density', '1')
    elif body['size'] == 'medium':
      geom_attr[3] = ('density', '2')
    elif body['size'] == 'small':
      geom_attr[3] = ('density', '4')

    geom_attr[1] = ('type', shape)
    geom_attr[0] = ('rgba', body['color_val'])
    geom_attr[6] = ('material', body['material'])
    if body['material'] == 'metal':
      # metal texture
      geom_attr[6] = ('material', body['color'] + '_metal')

    geom = ET.SubElement(body_node, 'geom')
    set_attribute(geom, geom_attr)
    # friction
    friction_joint_1[0] = ('name', 'obj{}_slide'.format(count))
    fr_1 = ET.SubElement(body_node, 'joint')
    set_attribute(fr_1, friction_joint_1)
    count += 1


def set_point_mass(root, worldbody, location_str):
  """Add actuated point-mass agent to the xml tree."""
  point_mass = ET.SubElement(worldbody, 'body')
  point_mass.set('name', 'point_mass')
  point_mass.set('pos', location_str)
  friction_joint_y = [('name', 'pm_joint_y'), ('type', 'slide'),
                      ('pos', '0 0 0'), ('axis', '0 1 0'),
                      ('range', '-10.3213 10.3'), ('damping', '0.5')]
  friction_joint_x = [('name', 'pm_joint_x'), ('type', 'slide'),
                      ('pos', '0 0 0'), ('axis', '1 0 0'),
                      ('range', '-10.3213 10.3'), ('damping', '0.5')]
  pm_geom_attr = [('name', 'pm'), ('type', 'sphere'), ('rgba', '1 1 1 1'),
                  ('size', '0.05'), ('contype', '1'), ('conaffinity', '0'),
                  ('density', '0.5')]
  wrist_link_geom = ET.SubElement(point_mass, 'geom')
  wrist_link_joint_x = ET.SubElement(point_mass, 'joint')
  wrist_link_joint_y = ET.SubElement(point_mass, 'joint')
  set_attribute(wrist_link_joint_x, friction_joint_x)
  set_attribute(wrist_link_joint_y, friction_joint_y)
  set_attribute(wrist_link_geom, pm_geom_attr)
  # actuators
  actuator = ET.SubElement(root, 'actuator')
  wl_joint_actuator_x = ET.SubElement(actuator, 'motor')
  wl_joint_actuator_y = ET.SubElement(actuator, 'motor')

  actuator_x_attr = [('joint', 'pm_joint_x'), ('ctrlrange', '-2.0 2.0'),
                     ('ctrllimited', 'true')]
  actuator_y_attr = [('joint', 'pm_joint_y'), ('ctrlrange', '-2.0 2.0'),
                     ('ctrllimited', 'true')]

  set_attribute(wl_joint_actuator_x, actuator_x_attr)
  set_attribute(wl_joint_actuator_y, actuator_y_attr)


def set_simple_gripper(root, worldbody):
  """Add actuated pushing end-factor to the xml tree."""
  wrist_link = ET.SubElement(worldbody, 'body')
  wrist_link.set('name', 'wrist_link')
  wrist_link.set('pos', '0 0.8 -0.15')

  friction_joint_y = [('name', 'wrist_joint_y'), ('type', 'slide'),
                      ('pos', '0 0 0'), ('axis', '0 1 0'),
                      ('range', '-10.3213 10.3'), ('damping', '0.5')]
  friction_joint_x = [('name', 'wrist_joint_x'), ('type', 'slide'),
                      ('pos', '0 0 0'), ('axis', '1 0 0'),
                      ('range', '-10.3213 10.3'), ('damping', '0.5')]
  wl_geom_attr = [('name', 'wl'), ('type', 'capsule'),
                  ('fromto', '0 -0.05 0 0 0.05 0'), ('size', '0.01'),
                  ('contype', '1'), ('conaffinity', '0'), ('density', '0.01')]

  wrist_link_geom = ET.SubElement(wrist_link, 'geom')
  wrist_link_joint_x = ET.SubElement(wrist_link, 'joint')
  wrist_link_joint_y = ET.SubElement(wrist_link, 'joint')
  set_attribute(wrist_link_joint_x, friction_joint_x)
  set_attribute(wrist_link_joint_y, friction_joint_y)
  set_attribute(wrist_link_geom, wl_geom_attr)
  gripper = ET.SubElement(wrist_link, 'body')
  gripper.set('name', 'gripper')
  gripper.set('pos', '0 0 0')
  geoms = [ET.SubElement(gripper, 'geom') for _ in range(3)]

  linking_arm = [('name', 'linking_arm'), ('type', 'capsule'),
                 ('fromto', '-0.1 0 0. +0.1 0 0'), ('size', '0.02'),
                 ('contype', '1'), ('conaffinity', '1'), ('density', '0.01')]
  left_arm = [('name', 'left_arm'), ('type', 'capsule'),
              ('fromto', '-0.1 0. 0 -0.1 0 -0.1'), ('size', '0.02'),
              ('contype', '1'), ('conaffinity', '1'), ('density', '0.01')]
  right_arm = [('name', 'right_arm'), ('type', 'capsule'),
               ('fromto', '0.1 0. 0 0.1 0 -0.1'), ('size', '0.02'),
               ('contype', '1'), ('conaffinity', '1'), ('density', '0.01')]

  for i, attr in enumerate([linking_arm, left_arm, right_arm]):
    set_attribute(geoms[i], attr)
  actuator = ET.SubElement(root, 'actuator')
  wl_joint_actuator_x = ET.SubElement(actuator, 'motor')
  wl_joint_actuator_y = ET.SubElement(actuator, 'motor')

  actuator_x_attr = [('joint', 'wrist_joint_x'), ('ctrlrange', '-2.0 2.0'),
                     ('ctrllimited', 'true')]
  actuator_y_attr = [('joint', 'wrist_joint_y'), ('ctrlrange', '-2.0 2.0'),
                     ('ctrllimited', 'true')]

  set_attribute(wl_joint_actuator_x, actuator_x_attr)
  set_attribute(wl_joint_actuator_y, actuator_y_attr)
