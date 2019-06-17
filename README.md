# CLEVR-Robot Environment

## Overview

The CLEVR-Robot environment is a reinforcement learning environment that aims to
provide a research platform for developing RL agents at the intersection of
vision, language, and continuous/discrete control.

The environment is inspired by the
[CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/) which is one of
the de facto standard datasets for visual-question-answering, and we integrate
part of its
[generation code](https://github.com/facebookresearch/clevr-dataset-gen/tree/master/question_generation)
to emit language description of the objects in the environment built on top of
the [MuJoCo](http://www.mujoco.org/) physics simulator.

## Preliminaries

The environment can contain up to 5 objects with customizable **color**,
**shape**, **size**, and **material** on a table top with 4 invisible planes at
4 cardinal directions which prevent the obejcts from leaving the table top. The
environment supports both state-based observation and image-based observation
(adjustable through arguments `obs_type`). The environment contains a scene graph or a world
state that keeps track of the objects and their locations.

In addition to observations, we also introduce textual descriptions of the
environment. These descriptions resemble questions in the CLEVR dataset, but
they do not actually need to be questions (this can also be adjusted through
templates). For clarity, it is instrumental to think of the descriptions as
vectors that can be *evaluated* on the state, which makes question-answering a
special case from this perspective. If you squint hard enough, you can make a
vague connection between answering the question and taking an "inner product"
between the question and the observation. Like CLEVR, we leverage 2
representations for these description: `text` and `program`. The former is just
a regular text such as *"There is a green rubber ball; is there a large red ball
to the left of it?"* and the `program` is a functional program that can be
exectued on the scene graph to yield an answer. The latter can be used for
providing the ground truth value to the text.

*Note: The full spectrum of variation may be too hard for existing RL agent that
uses pixel observation so in practice we advise starting with smaller diversity
for vision-based agents).*

## Usage

At a high level, the environment uses the standard OpenAI gym interface, but it
has many additional functionalities to bring language into the picture. The
environment is self-contained so it can be imported like a regular MuJoCo gym
environment, but it has OpenCV to handle some image processing pipeline.

```Python
env = ClevrEnv()
```
To take an action in this envrionment (which supports different action
parameterizations), we can sample a random action and use the step function.
We can also reset the environment which will randomly perturb the existing
objects, or sample a new set of objects by setting `new_scene_content=True` (This only
works if you are in the diverse objects setting and has no effect otherwise).

```Python
action = env.sample_random_action()
obs, reward, _, _ = env.step(action)
obs = env.reset()  # regular reset
obs = env.reset(new_scene_content=True) # sample new objects
```

`env.sample_random_action` differs from `env.action_space.sample` if the action
space of the environment changes dynamically (i.e. it is an option to push
each object direclty so the action space changes when number of objects
changes.)

### Language Descriptions

To get the descriptions of the scene, we can do:

```Python
description, full_description = env.get_description()
```

`description` is a list of string that describes the current scene and
`full_description` is a list of dictionaries. Each dictionary has the following
content:

*   `question`: this is the same as the text description
*   `program`: a piece of functional program that can be executed on the scene
    graph
*   `answer`: the value of the question on the current scene. (currently we are
    supporting boolean answers but other extensions should be easy)

Interanlly, this information is available at `env.descriptions` and
`env.full_descriptions`; however, because computing this at every step may be
expensive and unnecessary, we use lazy evaluation which only re-compute it when needed.
You may also opt to update them at every step by:

```Python
obs, reward, _, info = env.step(action, update_des=True)
current_descriptions = info['descriptions']
current_full_descriptions = info['full_descriptions']
```

At initialization and also at the reset, the environment sets a random question
and its program as the **goal** of the environment. The default reward of the
environment is a binary reward indicating if this goal has been reached (i.e.
the value of the description is 1). To get a random new goal and to set the goal
of the environment, we can do the following:

```Python
goal_text, goal_program = env.sample_goal()
env.set_goal(goal_text, goal_program)  # set the new goal
```

### Goal-conditioned Reinforcement Learning

Perhaps you have already noticed, this environment is designed with
goal-conditioned reinforcement learning in mind, where these language descriptions
effectively act as the goal or, perhaps more accurately, the **instruction**
about what to do for the agent. Given a instruction program, we can evaluate the
value of the program by:

```Python
answer = env.answer_question(goal_program)
```

More importantly, we can pass in an additional `goal` argument into the step
function such that the goal program passed in will overwrite the default reward
function:

```Python
obs, reward, _, _ = env.step(action, goal=goal_program)
```

Moreover, we can also record a set of instructions that are fulfilled as the
result of performing a action. Fulfilled here means that the values of these
descriptions are False before the interaction and become True as the result of
the action.

```Python
obs, reward, _, _ = env.step(action, record_achieved_goal=True)
achieved_goal_text = env.get_achieved_goals()
achieved_goal_program = env.get_achieved_goal_programs()
```

This functionality helps establish a causal relationship between the agent's action
and the changes induced.

Work in progress. More documentation is coming.

Disclaimer: This is not an official Google product.
