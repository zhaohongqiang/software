from mission.constants.config import PIPE_SEARCH_DEPTH
from mission.framework.combinators import *
from mission.framework.position import *
from mission.framework.task import *
from mission.framework.timing import *
from mission.opt_aux.aux import *
from mission.missions.buoys import Scuttle

import aslam

import shm, time, math
import numpy as n

from auv_math.math_utils import rotate

class Timeout(Task):
    def on_first_run(self, time, task, *args, **kwargs):
        self.timer = Timer(time)

    def on_run(self, time, task, *args, **kwargs):
        task()
        self.timer()
        if task.finished:
          self.finish()
        elif self.timer.finished:
          self.logw('Task timed out in {} seconds!'.format(time))
          self.finish()

Scan = Sequential(
  MoveYRough(1.5),
  MoveYRough(-3.0),
  MoveYRough(3.0),
  MoveYRough(-3.0),
  MoveYRough(1.5)
)

boundingBox = lambda pos: (pos - n.array([0.2, 0.2, 0.2]), pos + n.array([0.2, 0.2, 0.2]))

tolerance = n.array([0.03, 0.03, 0.03])

class TouchGuarded(Task):
  def on_run(self, subtask, sensor):
    subtask()
    if subtask.finished or not sensor.get():
      self.finish()

class AvoidYellow(Task):
  def on_first_run(self):
    self.heading        = shm.kalman.heading.get()
    self.red_buoy       = aslam.world.red_buoy.position()[:2]
    self.green_buoy     = aslam.world.green_buoy.position()[:2]
    self.yellow_buoy    = aslam.world.yellow_buoy.position()[:2]
    self.all_buoys      = [('red', self.red_buoy), ('green', self.green_buoy), ('yellow', self.yellow_buoy)]
    self.sorted_buoys   = sorted(self.all_buoys, key = lambda x: rotate(x[1], -self.heading)[1])
    self.logi('Sorted buoys (left-to-right): {}'.format([x[0] for x in self.sorted_buoys]))
    subtasks = []
    subtasks.append(MoveXRough(-0.5))
    subtasks.append(SmallXYZWiggle())
    if self.sorted_buoys[0][0] == 'yellow':
        # yellow buoy far left, go right
        subtasks.append(MoveYRough(1.0))
    elif self.sorted_buoys[1][0] == 'yellow':
        subtasks.append(MoveYRough(1.0))
    else:
        subtasks.append(MoveYRough(-1.0))
    subtasks.append(MoveXRough(1.0))
    center_buoy = n.array(self.sorted_buoys[1][1])
    center_buoy += n.array(rotate((2, 0), self.heading)) # 2m beyond center buoy
    subtasks.append(GoToPosition(center_buoy[0], center_buoy[1], depth=PIPE_SEARCH_DEPTH))
    self.subtask = Sequential(*subtasks)

  def on_run(self):
    self.subtask()
    if self.subtask.finished:
        self.finish()

class AllBuoys(Task):
  def possibleModes(self):
    if self.finished:
      return []

    return [
      Mode(name = 'ThreeBuoys', expectedPoints = 1400, expectedTime = 200),
      Mode(name = 'RedBuoy', expectedPoints = 400, expectedTime = 80),
      Mode(name = 'RedAndGreen', expectedPoints = 800, expectedTime = 120),
      Mode(name = 'Scuttle', expectedPoints = 600, expectedTime = 60)
      ]

  def desiredModules(self):
    return [shm.vision_modules.Buoys]

  def on_first_run(self, mode = 'ThreeBuoys'):

    self.loge('Mode: {}'.format(mode))

    delta_red = aslam.world.red_buoy.position() - aslam.sub.position()
    delta_red /= n.linalg.norm(delta_red)
    delta_red *= -1
    delta_green = aslam.world.green_buoy.position() - aslam.sub.position()
    delta_green /= n.linalg.norm(delta_green)
    delta_green *= -1
    delta_yellow = aslam.world.yellow_buoy.position() - aslam.sub.position()
    delta_yellow /= n.linalg.norm(delta_yellow)
    delta_yellow *= -1

    subtasks = []
    # subtasks.append(aslam.SimpleTarget(aslam.world.red_buoy, delta_red * 4))
    # subtasks.append(aslam.Orient(aslam.world.red_buoy))
    subtasks.append(MoveXRough(1.2))
    subtasks.append(Scan)
    #subtasks.append(MoveXRough(1.0))
    #subtasks.append(Scan)
    if mode == 'RedBuoy' or mode == 'RedAndGreen' or mode == 'ThreeBuoys':
      subtasks += [
        Timeout(20.0, aslam.Target(aslam.world.red_buoy, delta_red, tolerance, boundingBox(delta_red * 2), orient = True)),
        Timeout(5.0, TouchGuarded(MoveXRough(1.3), shm.gpio.wall_1)),
        MoveXRough(-2.0)
      ]

    if mode == 'RedAndGreen' or mode == 'ThreeBuoys':
      subtasks += [
        Timeout(20.0, aslam.Target(aslam.world.green_buoy, delta_green, tolerance, boundingBox(delta_green * 2), orient = True)),
        Timeout(5.0, TouchGuarded(MoveXRough(1.3), shm.gpio.wall_1)),
        MoveXRough(-2.0)
      ] 
    
    if mode == 'Scuttle' or mode == 'ThreeBuoys':
      subtasks += [
        Timeout(20.0, aslam.Target(aslam.world.yellow_buoy, delta_yellow, tolerance, boundingBox(delta_yellow * 2), orient = True)),
        GuardedTimer(10.0, Scuttle(), aslam.SimpleTarget(aslam.world.yellow_buoy, delta_yellow)),
        AvoidYellow()
      ]

    # subtasks.append(aslam.Orient(aslam.world.pipe_to_navigation))
    subtasks.append(Depth(0.5))
    subtasks.append(MoveXRough(2.0))
    self.subtask = Sequential(*subtasks)

  def on_run(self, mode = 'ThreeBuoys'):
    self.subtask()
    if self.subtask.finished:
      self.finish()
