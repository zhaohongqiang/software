from mission.framework.task import Task
from mission.framework.timing import Timer
from mission.framework.combinators import *
from mission.framework.helpers import *
from mission.framework.position import *
from mission.framework.movement import *
from auv_python_helpers.cameras import *

import shm, aslam, time, math, random, enum
import numpy as n

class State(enum.Enum):
  SEARCHING   = 0
  TARGETING   = 1
  PRERAMMING  = 2
  RAMMING     = 3

class Target(Task):
  def on_first_run(self, *args, **kwargs):
    self.last_obs = time.time()
    self.state = State.SEARCHING
    self.ram   = MoveX(1.2)
    self.left  = Concurrent(RelativeToInitialDepth(0.05, error = 0.02), MoveY(-0.25))
    self.right = Concurrent(RelativeToInitialDepth(-0.1, error = 0.02), MoveY(0.50))
    self.fwd   = MoveX(1.0)
    self.cc    = ConsistencyCheck(count = 8, total = 10)
    self.logv('Started!', copy_to_stdout = True)

  def switch_to(self, new_state):
    self.logv('Switching from state {} to state {}'.format(self.state, new_state), copy_to_stdout = True)
    self.state = new_state

  def sway_acquire(self):
    shm.navigation_settings.optimize.set(False) 

    if self.left.finished and self.right.finished:
      self.left  = Concurrent(RelativeToInitialDepth(0.1, error = 0.02), MoveY(-0.5))
      self.right = Concurrent(RelativeToInitialDepth(-0.1, error = 0.02), MoveY(0.5))
    elif self.left.finished:
      self.right()
    elif self.fwd.finished:
      self.left()
    else:
      self.fwd()

  def on_run(self, buoy, results):
    buoy_pos  = buoy.position()
    sub_pos   = aslam.sub.position()
    vision    = results.get()

    self.cc.add(vision.probability > 0.1)

    self.logv('{}'.format(self.state), copy_to_stdout = True)

    if vision.probability > 0.1:
      
      pixel_size = get_camera_pixel_size(vision)
      focal_length = get_camera_focal_length(vision)
      ctr_x, ctr_y = get_camera_center(vision)
      obj_x, obj_y = vision.center_x, vision.center_y
    
      # Sonar / camera simulation: calculate actual positional delta.
      # rad = vision.radius
      # rad = vision.area
      rad = math.sqrt(vision.area / math.pi)
      calc_angle_subtended = calc_angle(rad + ctr_x, ctr_x, pixel_size, focal_length)
      real_radius = 0.15
      calc_distance = real_radius / math.sin(calc_angle_subtended)

      # Camera-based heading / pitch / distance.
      calc_heading = calc_angle(obj_x, ctr_x, pixel_size, focal_length)
      calc_pitch = -calc_angle(obj_y, ctr_y, pixel_size, focal_length)
      # self.logv('Calc heading: {}, pitch: {}, distance: {}'.format(calc_heading, calc_pitch, calc_distance), copy_to_stdout = True)
      
      obs = aslam.Observation(aslam.sub, buoy, n.array([calc_heading, calc_pitch, calc_distance]), n.array([0.2, 0.2, 1.0]))
      # fake = aslam.Observation(aslam.sub, buoy, n.array([calc_pitch, calc_distance, calc_heading]), n.array([0.2, 0.2, 1.0]))
      # self.logv('Observational prior: {} Fake prior: {}'.format(obs.prior(), fake.prior()), copy_to_stdout = True)
      obs.apply()

    if self.state is State.SEARCHING:
  
      # Do some searching things!
      # aslam.sub.move_to(buoy_pos)

      if self.cc.check():
        self.left  = Concurrent(RelativeToInitialDepth(0.05, error = 0.02), MoveY(-0.25))
        self.right = Concurrent(RelativeToInitialDepth(-0.1, error = 0.02), MoveY(0.5))
        self.switch_to(State.TARGETING)
        self.start_time = time.time()

    if self.state is State.TARGETING:

      if not self.cc.check():
        self.switch_to(State.SEARCHING)
        return

      self.sway_acquire()

      uncertainty = n.linalg.norm(buoy.uncertainty())
      self.logv('Positional uncertainty: {}'.format(uncertainty), copy_to_stdout = True)

      if uncertainty < 0.1 and time.time() - self.start_time > 5.0:
        self.logv('Converged!', copy_to_stdout = True)
        cur_pos = sub_pos

        def preram():
          buoy_pos = buoy.position()
          diff    = buoy_pos - cur_pos
          heading = math.atan2(diff[1], diff[0])
          start   = buoy_pos - (n.array([math.cos(heading), math.sin(heading), 0.0]) * 1.0)
          return GoToPosition(start[0], start[1], heading = math.degrees(heading), depth = start[2])
        
        self.preram = preram
        self.switch_to(State.PRERAMMING)

    if self.state is State.PRERAMMING:
      task = self.preram()
      task()
      if task.finished:
        self.switch_to(State.RAMMING)

    if self.state is State.RAMMING:
      self.ram()
      if self.ram.finished:
        self.logv('Finished!', copy_to_stdout = True)
        #shm.navigation_settings.optimize.set(True) 
        self.finish()
    return
    if self.state is State.TARGETING:
      diff    = buoy_pos - sub_pos
      heading = math.atan2(diff[1], diff[0])
      pitch   = -math.atan2(diff[2], math.sqrt(diff[1] ** 2. + diff[0] ** 2.))
      shm.navigation_desires.heading.set(math.degrees(heading))
      # shm.navigation_desires.pitch.set(math.degrees(pitch))

Demo = lambda: Sequential(
  # GoToPosition(-1, 4, heading = 210, depth = 0.9),
  Target(buoy = aslam.world.buoy_a, results = shm.red_buoy_results),
  Depth(0.25),
  GoToPosition(-1, 4, heading = 210, depth = 0.9),
  Target(buoy = aslam.world.buoy_b, results = shm.green_buoy_results),
  Depth(0.25),
  GoToPosition(-1, 4, heading = 210, depth = 0.9),
  Target(buoy = aslam.world.buoy_c, results = shm.yellow_buoy_results)
  )
