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

"""Utils for loading relevant files."""

import os
import numpy as np
import six.moves.cPickle as pickle

# pre-generated questions
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
pregen_path = os.path.join(parent_dir,
                           'assets/pregenerated_data/all_question.pkl')
variable_input_pregen_path = os.path.join(
    parent_dir, 'assets/pregenerated_data/all_questions_variable_input.pkl')



def load_all_question(path=pregen_path):
  with open(path, 'rb') as f:
    pregen_content = pickle.load(f)
  questions = []
  for q in pregen_content:
    questions.append((q['question'], q['program']))
  return questions


def create_train_test_question_split():
  """Create random train/test split on pregenerated questions."""
  all_questions = load_all_question(pregen_path)
  all_questions_len = len(all_questions)

  indices = np.arange(all_questions_len)
  indices_train = indices[::3].astype(np.int32)
  indices_test = np.int32(list(set(indices) - set(indices_train)))
  aq = np.array(all_questions)

  all_questions_train = list(aq[indices_train])
  all_questions_test = list(aq[indices_test])
  return all_questions_train, all_questions_test


def create_systematic_generalization_split():
  """Create systematic generalization split on pregenerated questions."""
  questions = load_all_question(pregen_path)
  filtered_colors = ['red']
  filtered_direction = ['right', 'behind', 'front', 'left']
  test_questions = []
  train_questions = []
  for qp in questions:
    q, _ = qp
    q_len = len(q)
    c_exist = [c in q[:q_len // 3] for c in filtered_colors]
    d_exist = [d in q for d in filtered_direction]
    if np.int32(c_exist).sum() > 0 and np.int32(d_exist).sum() > 0:
      test_questions.append(qp)
    else:
      train_questions.append(qp)
  return train_questions, test_questions
