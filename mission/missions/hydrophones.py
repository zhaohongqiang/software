import collections
import math
import random

import numpy as np

from scipy.cluster.hierarchy import fclusterdata

import shm

from auv_math.math_utils import rotate
from misc.hydro2trans import Localizer
from mission.constants.config import HYDROPHONES_SEARCH_DEPTH
from mission.framework.task import Task
from mission.framework.helpers import get_sub_position, get_sub_quaternion
from mission.framework.position import GoToPosition
from shm.watchers import watcher

TRACK_MAG_THRESH = 3000
PINGER_FREQUENCY = 37500

DO_PARALLAX = False

# If True will listen sideways towards the pinger
# Only if not swaying.
PERP_TO_PINGER = True

# Yay we got the third channel to work ☺.
USE_THREE = True

# Only valid if USE_THREE is true.
FOLLOW_HEADING = True

# TODO Better abstraction of ping following strategies.

def get_clusterable(data):
  return np.array(data).reshape((len(data), 1))

class ThrusterSilencer(Task):
  def __init__(self):
    super().__init__()
    self.silence_time = None

  def in_silence(self):
    return not shm.settings_control.enabled.get()

  def schedule_silence(self, silence_time, silence_length):
    self.cancel_silence()
    self.silence_time = silence_time
    self.silence_length = silence_length

  def cancel_silence(self):
    self.silence_time = None
    shm.settings_control.enabled.set(1)

  def on_run(self):
    if self.silence_time is None:
      return

    if self.this_run_time - self.silence_time > self.silence_length:
      shm.settings_control.enabled.set(1)
      self.silence_time = None

    elif self.this_run_time > self.silence_time:
      shm.settings_control.enabled.set(0)

class FindPinger(Task):
  def on_first_run(self):
    shm.hydrophones_settings.track_frequency_target.set(PINGER_FREQUENCY)
    shm.hydrophones_settings.track_magnitude_threshold.set(TRACK_MAG_THRESH)
    shm.hydrophones_settings.track_cooldown_samples.set(150000)

    shm.navigation_settings.position_controls.set(1)
    shm.navigation_settings.optimize.set(0)

    self.localizer = Localizer(PINGER_FREQUENCY)

    self.hydro_watcher = watcher()
    self.hydro_watcher.watch(shm.hydrophones_results_track)

    self.time_since_last_ping = self.this_run_time

    self.pinger_positions = collections.deque(maxlen=7)

    self.PINGS_LISTEN = 5
    self.CONSISTENT_PINGS = 3

    self.listens = 0
    self.queued_moves = []
    self.pinger_found = False

    self.silencer = ThrusterSilencer()

    self.available_depths = set([HYDROPHONES_SEARCH_DEPTH,
                                 HYDROPHONES_SEARCH_DEPTH + 0.4,
                                 HYDROPHONES_SEARCH_DEPTH - 0.4])
    self.last_depth = None

    self.observe_from(get_sub_position())

  def observe_from(self, position, heading=None):
    self.logi("Moving to %0.2f %0.2f to observe some more." % \
              (position[0], position[1]))

    depth = random.choice(list(self.available_depths - set([self.last_depth])))
    self.last_depth = depth
    self.motion_tasks = GoToPosition(position[0], position[1],
                                     heading=heading, depth=depth)
    self.pings_here = 0
    self.pings = collections.deque(maxlen=self.PINGS_LISTEN)

    self.listens += 1

    self.silencer.cancel_silence()

  def on_run(self):
    self.motion_tasks()

    self.silencer()

    # TODO Better way to filter out bad pings when thrusters are running.
    if not self.hydro_watcher.has_changed() or \
       not self.motion_tasks.finished:
      # TODO Do something here if too much time has passed since last ping.
      return

    if self.pinger_found:
      self.finish()
      return

    self.time_since_last_ping = self.this_run_time
    self.pings_here += 1

    # TODO Will this be long after the watcher fired?
    # Need to ensure that there is little delay.
    results = shm.hydrophones_results_track.get()
    kalman = shm.kalman.get()

    phases = (results.diff_phase_x, results.diff_phase_y)
    self.logi("Got " + str(phases))

    in_silence = self.silencer.in_silence()

    self.silencer.schedule_silence(self.this_run_time + 0.9, 0.4)

    if not in_silence:
      return

    self.pings.append(results.diff_phase_x)

    if len(self.pings) < 2:
      return

    data = get_clusterable(self.pings)
    clusters = fclusterdata(data, 0.1, criterion="distance")
    counted = collections.Counter(clusters)

    best_cluster, n_best = max(counted.items(), key=lambda item: item[1])
    if n_best >= self.CONSISTENT_PINGS:
      avg_phase = sum([self.pings[i] for i, cluster_num in enumerate(clusters) if cluster_num == best_cluster]) / n_best

      self.logi("Found nice phase %f in %s" % (avg_phase, str(self.pings)))

      if FOLLOW_HEADING:
        # TODO Use average x and y phases here.
        heading, elevation = self.localizer.get_heading_elevation(*phases)

        if elevation > 45:
          move_distance = 4
        elif elevation > 30:
          move_distance = 2
        else:
          move_distance = 1

        sub_pos = get_sub_position(kalman)
        direction = rotate((1, 0), kalman.heading + heading)
        observe_position = sub_pos + np.array((direction[0], direction[1], sub_pos[2])) * move_distance
        self.logi("Heading is %f, Elevation is %f" % (heading, elevation))
        self.observe_from(observe_position, kalman.heading + heading)

      elif not FOLLOW_HEADING:
        sub_pos = get_sub_position(kalman)
        sub_quat = get_sub_quaternion(kalman)

        self.localizer.add_observation(phases, sub_pos, sub_quat)

        # For now, use the estimate from only 2 transducers.
        est_pinger_pos = self.localizer.compute_position()[USE_THREE]
        self.pinger_positions.append(est_pinger_pos)

        self.logi("We think the pinger is at %s" % str(est_pinger_pos))
        self.logi("All estimated positions: %s" % str(self.pinger_positions))

        if len(self.pinger_positions) > 5:
          data = np.array(self.pinger_positions)
          clusters = fclusterdata(data, 0.5, criterion="distance")
          counted = collections.Counter(clusters)
          best_cluster, n_best = max(counted.items(), key=lambda item: item[1])
          if n_best > 3:
            avg_position = sum([np.array(self.pinger_positions[i]) for i, cluster_num in enumerate(clusters) if cluster_num == best_cluster]) / n_best
            self.logi("Found pinger at %0.2f %0.2f!" % (avg_position[0], avg_position[1]))
            self.observe_from(avg_position)
            self.pinger_found = True
            return

        if self.listens <= 3:
          # TODO Change heading in an optimal fashion.
          self.observe_from(sub_pos, heading=(kalman.heading + 30) % 360)
        else:
          if not self.queued_moves:
            to_pinger = est_pinger_pos[:2] - sub_pos[:2]
            distance = np.linalg.norm(to_pinger)
            direction = to_pinger / distance

            move_dist = 3
            if self.listens > 10 and distance < 3:
              move_dist = distance

            new_pos = sub_pos[:2] + direction * move_dist
            heading = math.atan2(direction[1], direction[0])

            if DO_PARALLAX:
              perp = np.array(rotate(direction, 90))
              self.queued_moves.extend([(new_pos, heading), (new_pos + 2 * perp, heading), (new_pos - 2 * perp, heading)])

            else:
              if PERP_TO_PINGER:
                heading += 90
              self.queued_moves.extend([(new_pos, heading+30), (new_pos, heading-30), (new_pos, heading)])

          next_pos, heading = self.queued_moves.pop(0)
          self.observe_from(next_pos, heading)
