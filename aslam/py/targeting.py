from collections import namedtuple

from mission.framework.helpers import *
from mission.framework.search import *
from mission.framework.movement import *
from mission.framework.position import *
from mission.framework.task import *
from termcolor import colored

import aslam.py.core as aslam

import shm, time, math, enum, random
import numpy as n

class TargetState(enum.Enum):
  PREINITIALIZING = 0
  INITIALIZING    = 1
  SWAYSEARCHING   = 2
  SPIRALSEARCHING = 3
  ACQUIRING       = 4
  FINALIZING      = 5

def within(x, y):
  return all([x[i] < y[i] for i in range(len(x))])

def center(box):
  return (box[0] + box[1]) / 2

def all_corners():
  return [n.array([-1.0, -1.0, -1.0]),
          n.array([-1.0, -1.0,  1.0]),
          n.array([-1.0,  1.0, -1.0]),
          n.array([-1.0,  1.0,  1.0]),
          n.array([1.0,  -1.0, -1.0]),
          n.array([1.0,  -1.0,  1.0]),
          n.array([1.0,   1.0, -1.0]),
          n.array([1.0,   1.0,  1.0])]

class Orient(Task):
  def on_run(self, obj, pitch = False):
    dlt  = obj.position() - aslam.sub.position()
    h, p = math.atan2(dlt[1], dlt[0]), math.atan2(-dlt[2], (dlt[0] ** 2. + dlt[1] ** 2.) ** 1./2)
    heading_task = Heading(math.degrees(h), deadband = 5)
    heading_task()

    if pitch:
      pitch_task   = Pitch(math.degrees(p), deadband = 5)
      pitch_task()

      if heading_task.finished and pitch_task.finished:
        self.finish()

    else:
      if heading_task.finished:
        self.finish()

class SimpleTarget(Task):
  def on_run(self, obj, rel):
    des_pos = obj.position() + rel
    subtask = GoToPosition(des_pos[0], des_pos[1], depth = des_pos[2])
    subtask()
    if subtask.finished:
      self.finish()

# By default, will start at initial depth and sweep area.

class Target(Task):
  '''
  Generic targeting task.

  :param object                 : Object you wish to target
  :param finalPosition          : Final desired position
  :param finalSubTolerance      : Tolerance in sub error for final position
  :param finalObjectTolerance   : Tolerance in object uncertainty for final position
  :param observationBoundingBox : Bounding box to traverse to observe the object
  :param orient                 : If true, orient towards the target as we observe it
  :param searchRadius           : Radius from object position to begin searching
  :param searchWidth            : Width of box search
  :param searchStride           : Stride of box search
  :param searchTolerance        : Tolerance in object certainty to switch from search to observation
  '''

  def switch_state(self, new_state):
    self.logi('Targeting of object {} switching from state {} to state {}'.format(self.obj.name, self.state, new_state))
    self.state = new_state

  def select_corner(self):
    if self.corners_to_visit == []:
      self.corners_to_visit = all_corners()
      random.shuffle(self.corners_to_visit)
    self.target_corner = self.corners_to_visit.pop()

  def on_first_run(self, obj, finalPosition, finalTolerance, observationBoundingBox, orient = True, searchRadius = 5.0, approachRadius = 2.0, searchWidth = 4.0, searchStride = 1.0, searchTolerance = n.array([1.0, 1.0, 1.0])):
    self.start_time = time.time()
    self.corners_to_visit = []
    self.obj = obj
    self.state = None
    self.switch_state(TargetState.INITIALIZING)
    self.logi('Target initialized. Attempting to position sub at position {} using object {}. Start time: {}'.format(
      '[N {} E {} D {}]'.format(*finalPosition),
      obj.name,
      self.start_time
      ))
    shm.navigation_settings.position_controls.set(True)
    shm.navigation_settings.optimize.set(True)
    self.logi('Enabled positional control and heading optimization.')

    obj_pos = obj.position()
    sub_pos = aslam.sub.position()
    delta   = obj_pos - sub_pos
    if n.linalg.norm(delta) < approachRadius:
      self.logi('Submarine started within approach box. Will move to outside of search box prior to searching')
      self.switch_state(TargetState.PREINITIALIZING)
      options = [obj_pos + n.array([math.cos(h) * approachRadius, math.sin(h) * approachRadius, 0.0]) for h in n.arange(0, 2 * math.pi, math.pi / 32)]
      self.presearch_position = min(options, key = lambda pos: n.linalg.norm(sub_pos - pos))
      self.logi('Selected presearch position: {}'.format('[N {} E {} D {}]'.format(*self.presearch_position)))

  def on_run(self, obj, finalPosition, finalTolerance, observationBoundingBox, orient = True, searchRadius = 6.0, approachRadius = 4.0, searchWidth = 4.0, searchStride = 1.0, searchTolerance = n.array([0.5, 0.5, 0.5])):
    sub_pos = aslam.sub.position()
    obj_unc = obj.uncertainty()
    obj_pos = obj.position() if n.linalg.norm(obj_unc) < 1. else obj.initial()[:3]
    delta   = obj_pos - sub_pos

    self.last_unc_improved = True if not hasattr(self, 'last_unc') else np.linalg.norm(obj_unc) < self.last_unc
    self.last_unc = np.linalg.norm(obj_unc)
    if self.last_unc_improved:
      self.last_unc_improvement = time.time()

    observationBoundingBox = (observationBoundingBox[0] + obj_pos, observationBoundingBox[1] + obj_pos)
    finalPosition = finalPosition + obj_pos

    if self.state is TargetState.PREINITIALIZING:
      gtp = GoToPosition(self.presearch_position[0], self.presearch_position[1], depth = self.presearch_position[2])
      gtp()

      if gtp.finished:
        self.logi('Submarine reached initial presearch position, initiating initialization')
        self.switch_state(TargetState.INITIALIZING)
        return

    # Initializing: move towards where we think the object is.
    if self.state is TargetState.INITIALIZING:
      GoToPosition(obj_pos[0], obj_pos[1], depth = obj_pos[2])()

      if n.linalg.norm(delta) < approachRadius:
        shm.navigation_settings.optimize.set(False)
        self.logi('Submarine within search bounding box, orienting towards object and initiating sway search')
        self.sway_search = Sequential(Orient(obj), SwaySearch(searchWidth, searchStride))
        self.sway_search_start = time.time()
        self.switch_state(TargetState.SWAYSEARCHING)
        return

    # Searching: sweep the area for the object.
    elif self.state is TargetState.SWAYSEARCHING:
      Depth(obj_pos[2])()
      self.sway_search()

      if within(obj_unc, searchTolerance) and time.time() - self.last_unc_improvement < 0.5:
        self.acq_start_unc = obj_unc
        self.acq_start_time = time.time()
        self.switch_state(TargetState.ACQUIRING)
        self.select_corner()
        return

      if n.linalg.norm(delta) > searchRadius and time.time() - self.sway_search_start > 5:
        self.logi('Left search area without finding object. Moving to object position and initiating spiral search')
        self.switch_state(TargetState.SPIRALSEARCHING)
        self.spiral_search = Sequential(GoToPosition(obj_pos[0], obj_pos[1], depth = obj_pos[2]), SpiralSearch())
        return

    elif self.state is TargetState.SPIRALSEARCHING:
      self.spiral_search()

      if within(obj_unc, searchTolerance) and time.time() - self.last_unc_improvement < 0.5:
        self.acq_start_unc = obj_unc
        self.last_acq_improvement = time.time()
        self.switch_state(TargetState.ACQUIRING)
        self.select_corner()
        return

    elif self.state is TargetState.ACQUIRING:

      if time.time() - self.last_unc_improvement > 5:
        self.logi('Object uncertainty unimproved in 5s of acquisition, switching to spiral search')
        self.spiral_search = SpiralSearch()
        self.switch_state(TargetState.SPIRALSEARCHING)

      if within(obj_unc, finalTolerance):
        self.logi('Object uncertainty within final tolerance, finalizing position')
        self.switch_state(TargetState.FINALIZING)
        return

      if not within(obj_unc, searchTolerance):
        self.logi('Object positional uncertainty increased, switching to spiral searching')
        self.spiral_search = SpiralSearch()
        self.switch_state(TargetState.SPIRALSEARCHING)
        return

      bb_del = observationBoundingBox[1] - observationBoundingBox[0]
      bb_ctr = center(observationBoundingBox)
      target = bb_ctr + (bb_del * self.target_corner / 2.0)
      GoToPosition(target[0], target[1], depth = target[2])()

      if orient:
        Orient(obj)()

      if n.linalg.norm(target - sub_pos) < 0.1:
        self.logi('Switching corners')
        self.select_corner()

    elif self.state is TargetState.FINALIZING:
      gtp = GoToPosition(finalPosition[0], finalPosition[1], depth = finalPosition[2])
      gtp()

      if orient:
        self.orient = Orient(obj)
        self.orient()

      if (self.orient.finished if orient else True) and gtp.finished:
        self.logi('Within final tolerance and oriented, finishing!')
        self.finish()
