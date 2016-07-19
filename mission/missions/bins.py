from mission.constants.config import BINS_DROP_ALTITUDE, \
                                     BINS_CAM_TO_ARM_OFFSET, \
                                     BINS_PICKUP_ALTITUDE, \
                                     BINS_SEARCH_DEPTH

from mission.framework.actuators import FireActuator
from mission.framework.combinators import Sequential, MasterConcurrent
from mission.framework.helpers import get_camera_center, ConsistencyCheck, get_sub_position
from mission.framework.movement import Depth, Heading, VelocityX, VelocityY, Roll, \
                                       RelativeToInitialDepth
from mission.framework.position import MoveX, MoveY, GoToPosition
from mission.framework.primitive import Log, FunctionTask
from mission.framework.targeting import DownwardTarget
from mission.framework.task import Task
from mission.framework.timing import Timer, Timed
from mission.opt_aux.aux import *

import aslam
import numpy as n

FAST_RUN = False
#may simply drop markers into open bin if running out of time

# ^ Should be an optimal mission runner mode!

import shm

bin1 = shm.bin1
bin2 = shm.bin2

GIVEN_DISTANCE = 2 # meters
BIN_CONFIDENCE = .7

PERCENT_GIVEN_DISTANCE_SEARCH = .2 # percent of given distance to back up and check
SEARCH_SIDE_DISTANCE = 3 # meters side to side
SEARCH_ADVANCE_DISTANCE = 1 # meters to advance with each zigzag

def check_uncovered():
    """Actually, this is really rough, this function should probably store the sub's current postion, move the sub to a position where both bins are visible, and then return to the original position"""
    return not bin1.covered and not bin2.covered

class OptimizableBins(Task):
  def possibleModes(self):
    if self.finished: return []

    return [
      Mode(name = 'Full', expectedPoints = 2300, expectedTime = 200)
    ]

  def desiredModules(self):
    return [shm.vision_modules.Bins]

  def on_first_run(self, mode):
    self.subtask = \
      Sequential(
        aslam.SimpleTarget(aslam.world.bin_one, n.array([0., 0., -2.])),
        BinsTask()
      )
  
  def on_run(self, mode):
    self.subtask()
    if self.subtask.finished:
      self.finish()

class BinsTask(Task):
    """Drops markers into target bin

    Current setup:
    Search for bins
    Center over target bin (assume covered bin initially)
    Try two times to remove lid
    If succeed, center over bin and drop markers in
        If fail twice, switch target bin to uncovered bin and drop markers

    Start: near bins (used pinger to locate), any position
    Finish: centered over target bin, both markers dropped
    """
    def on_first_run(self, *args, **kwargs):
        self.logi("Starting BinsTask task")
        self.init_time = self.this_run_time

        # TODO Do a search for the bin after uncovering.
        self.tasks = Sequential(Depth(BINS_SEARCH_DEPTH),
                                SearchBinsTask(bin1), IdentifyBins(bin1),
                                FunctionTask(lambda: self.set_heading(shm.kalman.heading.get())),
                                UncoverBin(),
                                MasterConcurrent(CheckBinsInSight(bin2), Depth(BINS_SEARCH_DEPTH)),
                                IdentifyBins(bin2, heading=lambda: self.bins_heading),
                                DiveAndDropMarkers())

    def set_heading(self, heading):
        self.bins_heading = heading

    def on_run(self):
        # self.logv("running bt")
        if self.tasks.finished:
            self.finish()

        self.tasks()
            # SearchBinsTask()

        # if not self.try_removal:
        #     self.try_removal = TwoTries()
        # elif not self.try_removal.has_ever_finished:
        #     self.try_removal()

        # #completed and succeded in removing lid
        # elif self.try_removal.success:
        #     self.drop_task = DiveAndDropMarkers()

        # #did not remove lid after two tries, search and center over uncovered bin
        # else:
        #     FAST_RUN = True
        #     self.search_and_identify = Sequential(SearchBinsTask(), IdentifyBins())
        #     self.drop_task = DiveAndDropMarkers()

        # #drop task either created after successfully uncovered bin OR doing simple run
        # if self.search_and_identify.has_ever_finished and self.drop_task and not self.drop_task.has_ever_finished:
        #     self.drop_task()

        # if self.drop_task.has_ever_finished:
        #     self.finish()

    def on_finish(self):
        self.logi("Bins completed!")
        self.logv('BinsTask task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

class SearchBinsTask(Task):
    """Uses SearchBinsTaskHelper in a MasterConcurrent with CheckBinsInSight"""
    def on_first_run(self, bin_group, *args, **kwargs):
        self.logi("Looking for bins...")
        self.logv("Starting SearchBinsTask")
        self.init_time = self.this_run_time
        # self.task = MasterConcurrent(SearchBinsTaskHelper(), CheckBinsInSight())
        self.task = MasterConcurrent(CheckBinsInSight(bin_group), SearchBinsTaskHelper())
        self.task()
    def on_run(self):
        self.task()
        if self.task.has_ever_finished:
            VelocityX(0.0)()
            self.finish()
    def on_finish(self):
        self.logi("Found bin!")
        self.logv('SearchBins task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

class SearchBinsTaskHelper(Task):
    """Looks around for bins, either covered or uncovered depending on FAST_RUN

    Suggestion: look around until the probabilities are > 0 for each bin? (see above for values imported from shm)

    Start: near bins (used pinger to locate), any position
    Finish: both bins visible in downward cam
    """
    def on_first_run(self, *args, **kwargs):
        self.logv("Starting SearchHelperTask task")
        self.init_time = self.this_run_time
        self.count = 0
#        self.zigzag = Sequential(Finite(MoveY(-SEARCH_SIDE_DISTANCE * 2)),
#                                  Finite(MoveX(SEARCH_ADVANCE_DISTANCE)),
#                                  Finite(MoveY(SEARCH_SIDE_DISTANCE * 2)),
#                                  Finite(MoveX(SEARCH_ADVANCE_DISTANCE)))

        # self.zigzag = Sequential(MoveY(-SEARCH_SIDE_DISTANCE * 2),
        #                         MoveX(SEARCH_ADVANCE_DISTANCE),
        #                         MoveY(SEARCH_SIDE_DISTANCE * 2),
        #                         MoveX(SEARCH_ADVANCE_DISTANCE))

        self.zigzag = VelocityX(0.5)
        self.stop = VelocityX(0.0)

        self.zigzag()

    def on_run(self):
        # self.logv("in sbth!")
        self.zigzag()

        if self.zigzag.has_ever_finished:
            self.count += 1
            if self.cycleCount < ((GIVEN_DISTANCE * PERCENT_GIVEN_DISTANCE_SEARCH * 2) / SEARCH_ADVANCE_DISTANCE) + 1:
                self.zigzag()
            else:
                self.loge("Failed to find bins")
                self.finish()

    def on_finish(self):
        self.stop()
        self.logv('SearchHelperTask task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

class CheckBinsInSight(Task):
    """ Checks if both bins are in sight of the cameras
    Used in SearchBinsTask as MasterConcurrent's end condition"""
    def on_first_run(self, *args, **kwargs):
        self.logv("Checking if bins in sight")
        self.init_time = self.this_run_time

    def on_run(self, bin_group):
        bin_results = bin_group.get()
        # self.logv("running cbis")
        if bin_results.p > 0.1:
            self.logv("probabilities work!")
            self.finish()

    def on_finish(self):
        VelocityX(0.0)()
        self.logv('CheckBinsInSight task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

class IdentifyBins(Task):
    """Identifies which bin to drop markers into, centers over it

    Start: Both bins visible in downward cam
    Finish: Centered over chosen bin
    """
    def on_first_run(self, bin_group, heading=None, *args, **kwargs):
        self.logi("Centering over bins...")
        self.logv("Starting IdentifyBins task")
        self.task = DownwardTarget(px=0.0025, py=0.0025)
        self.align_checker = ConsistencyCheck(6, 6)
        # TODO start alignment task.
        self.init_time = self.this_run_time

        self.bin_group = bin_group

        if bin1.covered == FAST_RUN:
            self.target_bin = bin1
        else:
            self.target_bin = bin2

    def on_run(self, bin_group, heading=None):
        self.bin1_results = self.bin_group.get()
        target = get_camera_center(self.bin1_results)

        # TODO Increase deadband as time increases.
        self.task((self.bin1_results.x, self.bin1_results.y), target=target, deadband=(25, 25), valid=lambda: self.bin1_results.probability > 0.0)

        if self.task.finished:
            if heading is None:
              target_heading = shm.kalman.heading.get() + self.bin1_results.angle
            else:
              target_heading = heading()

            align_task = Heading(target_heading, deadband=0.5)
            align_task()
            if self.align_checker.check(align_task.finished):
                VelocityX(0)()
                VelocityY(0)()
                self.finish()
        else:
            self.align_checker.clear()

    def on_finish(self):
        self.logi("Centered!")
        self.logv('IdentifyBins task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))


class TwoTries(Task):
    """Keeps track of how many times sub tried to uncover bin, changes variable FAST_RUN to True if sub was unable to take off bin cover

    Note: tried to keep logic as generic as possible, with self.attempt_task and self.check_done so can be reused for other missions

    Start: centered over covered bin, no markers dropped
    Finish: centered over covered bin, either both or no markers dropped
    """
    def on_first_run(self, *args, **kwargs):
        self.logv("Starting TwoTries task")
        self.init_time = self.this_run_time

        self.attempt_task = UncoverBin()
        self.check_done = check_uncovered

        self.success = False
        self.tries_completed = 0

    def on_run(self):
        if self.tries_completed==0:
            if not self.attempt_task.has_ever_finished:
                self.attempt_task()
            else:
                if self.check_done():
                    self.success = True
                    self.finish()
                else:
                    self.tries_completed = 1
                    self.attempt_task = UncoverBin()
        else: #one completed try, one try left
            if not self.attempt_task.has_ever_finished:
                self.attempt_task()
            else:
                self.success = self.check_done()
                self.finish()

    def on_finish(self):
        self.logv('TwoTries task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

class UncoverBin(Task):
    """Uses arm to remove bin cover (will most likely be Sequential?)

    Start: centered over covered bin, no markers dropped
    Finish: centered over now-uncovered bin
    """
    def on_first_run(self, slide_direction="right", *args, **kwargs):
        self.logi("Starting UncoverBin task")
        self.init_time = self.this_run_time

        if slide_direction == "right":
          factor = 1
        elif slide_direction == "left":
          factor = -1
        else:
          raise Exception("Invalid slide direction %s for UncoverBin" % slide_direction)

        self.initial_depth = shm.kalman.depth.get()
        self.set_depth = Depth(self.initial_depth + shm.dvl.savg_altitude.get() - BINS_PICKUP_ALTITUDE, deadband=0.01)
        self.initial_position = get_sub_position()

        self.pickup = MasterConcurrent(Sequential(Timer(2.0), Roll(-30 * factor, error=90)), VelocityY(1.0 * factor))
        self.drop = Sequential(Roll(90 * factor), Timer(1.0), Roll(120 * factor, error=40), Timer(0.5), Roll(90 * factor),
                               Timer(0.3), Roll(45 * factor), Timer(0.5), Roll(0))
        #self.return_to_bin = GoToPosition(lambda: self.initial_position[0], lambda: self.initial_position[1], depth=self.initial_depth)
        self.return_to_bin = Sequential(Timed(VelocityY(-1.0 * factor), 2.0), VelocityY(0.0), MoveX(-BINS_CAM_TO_ARM_OFFSET))
        self.tasks = Sequential(
                       MoveY(-0.65 * factor),
                       MoveX(BINS_CAM_TO_ARM_OFFSET),
                       self.set_depth,
                       Roll(60 * factor),
                       self.pickup,
                       VelocityY(0.0),
                       RelativeToInitialDepth(-1.0),
                       Timed(VelocityY(1.0 * factor), 1.0),
                       VelocityY(0.0),
                       self.drop,
                       Timed(VelocityY(-1.0 * factor), 1.0),
                       self.return_to_bin
                     )

    def on_run(self, *args):
        self.tasks()
        if self.tasks.finished:
            self.finish()

    def on_finish(self):
        self.logv('UncoverBin task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))

DropMarkersTogether = lambda: \
  Sequential(MoveY(-0.2), FireActuator("both_markers", 0.4),
  Log("Fired both droppers!"), MoveY(0.2))

DropMarkersSeperate = lambda: \
  Sequential(MoveY(-0.2), Log("Fired right dropper!"), MoveY(0.4),
  FireActuator("left_marker", 0.1), Log("Fired left dropper!"), MoveY(-0.2))

DropMarkers = DropMarkersTogether

class DiveAndDropMarkers(Task):
    """Drops markers into target bin

    Will need to lower self towards bin for better dropping accuracy

    Start: centered over target bin, no markers dropped
    Finish: centered over target bin, both markers dropped
    """
    def on_first_run(self, *args, **kwargs):
        self.logi("Starting DropMarkers task")
        self.init_time = self.this_run_time

        self.initial_depth = shm.kalman.depth.get()
        self.set_depth = Depth(self.initial_depth + shm.dvl.savg_altitude.get() - BINS_DROP_ALTITUDE, deadband=0.01)
        self.return_depth = Depth(self.initial_depth, deadband=0.01)

        self.timer = Timer(2.0)
        # TODO More accurate marker to bin alignment (Use the marker's position on the sub).
        self.seq = Sequential(self.set_depth, DropMarkers(), self.return_depth)

        self.timer()

    def on_run(self):
        self.seq()
        if self.seq.has_ever_finished:
            self.finish()

    def on_finish(self):
        self.logi("Dropped markers!")
        self.logv('DiveAndDropMarkers task finished in {} seconds!'.format(
            self.this_run_time - self.init_time))
