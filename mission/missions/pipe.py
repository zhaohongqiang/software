#!/usr/bin/env python3.4

from mission.constants.config import PIPE_SEARCH_DEPTH, PIPE_FOLLOW_DEPTH
from mission.framework.task import Task
from mission.framework.primitive import *
from mission.framework.combinators import Sequential
from mission.framework.movement import VelocityX, VelocityY, Depth, RelativeToCurrentHeading
from mission.framework.position import MoveX
from mission.framework.search import SearchFor, SwaySearch
from mission.framework.targeting import DownwardTarget, PIDLoop
from mission.framework.helpers import get_camera_center, ConsistencyCheck
from mission.framework.timing import *

from mission.opt_aux.aux import *
#from mission.framework.visiond import VisionModule

import time
import shm
import math

TORPEDO_OFFSET = -90

BuoyFail, WireFail = False, False

class Search(Task):
    def on_first_run(self, timeout=45):
        self.surge_time = time.time()
        self.sway_time = time.time() - 3
        self.start = time.time()
        self.surge = VelocityX()
        self.sway = VelocityY()
        self.sway_speed = .4
        self.swaying = True
        self.surging = False
        self.begin_search = time.time()

        self.sway(self.sway_speed)
        print("IN PIPE")

    def on_run(self, timeout=30):
        print("Searching For Pipe")
        if not self.surging:
            self.surge_time = time.time()

        if not self.swaying:
            self.sway_time = time.time()

        if self.this_run_time - self.sway_time > 6:
            self.sway(0)
            self.sway_speed *= -1
            self.swaying=False
            self.surge(.25)
            self.surging = True
        
        if self.this_run_time - self.surge_time > 3:
            self.surge(0)
            self.surging=False
            self.sway(self.sway_speed)
            self.swaying = True

        if shm.pipe_results.heuristic_score.get() < 3000:# or shm.pipe_results.rectangularity.get() < .5:
            self.start = time.time()

        if self.this_run_time - self.start > .5 or self.this_run_time - self.begin_search > timeout:
            self.surge(0)
            self.sway(0)
            self.finish()

class SurgeToPipe(Task):
    def on_first_run(self):
        self.start = time.time()
        self.surge = VelocityX(.3)
        self.count = time.time()

        self.surge()
    def on_run(self):
        if shm.pipe_results.heuristic_score.get() < 3000:
            self.start = time.time()

        if self.this_run_time - self.start > .4 or self.this_run_time - self.count > 10:
            self.surge(0)
            self.finish()

class SwayToPipe(Task):
    def on_first_run(self):
        self.start = time.time()
        self.surge = VelocityX(.2)
        self.sway = VelocityY(.2)

        self.surge()
        self.sway()
    def on_run(self):
        if shm.pipe_results.heuristic_score.get() < 3000:
            self.start = time.time()

        if self.this_run_time - self.start > .6:
            self.surge(0)
            self.sway(0)
            self.finish()

class ToBuoy(Task):
    def on_first_run(self, timeout=10):
        self.start = time.time()
        self.counter = time.time()

    def on_run(self, timeout=10):
        print("Looking for buoy")
        if self.this_run_time - self.counter > timeout:
            self._finish()
            return

        if not (shm.red_buoy_results.area.get() > 2000):
            self.start = time.time()

        if self.this_run_time - self.start > .7:
            self.finish()

class ToTorpedoes(Task):
    def on_first_run(self):
        self.start = time.time()
        self.counter = time.time()

    def on_first_run(self):
        
        if self.this_run_time - self.counter > 35:
            self._finish()
            return
        
        if shm.torpedo_results.target_center_x.get() < 1:
            self.start = time.time()

        if self.this_run_time - self.start > 1.7:
            self.finish()

class ToBins(Task):
    def on_first_run(self):
        self.start = time.time()
        self.shmstuff = [shm.shape_banana.p, shm.shape_bijection.p, shm.shape_lightning.p, shm.shape_soda.p, shm.shape_handle.p]
        self.count = time.time()

    def on_run(self):
        if not any(map(lambda x: x.get() > .5, self.shmstuff)):
            self.start = time.time()

        if self.this_run_time - self.start > 1 or self.this_run_time - self.count > 20:
            self.finish()

class ToWire(Task):
    def on_first_run(self):
        self.start = time.time()
        #shm.desires.depth.set(2.4)
        self.counter = time.time()

    def on_run(self):
        if self.this_run_time - self.counter > 28:
            self.finish()
            global WireFail
            WireFail = True
            return

        print("Going to wire")
        if shm.wire_results.area.get() < 18000:
            self.start = time.time()

        if self.this_run_time - self.start > 1:
            self.finish()

class center(Task):
    def update_data(self):
        self.pipe_results = shm.pipe_results.get()

    def on_first_run(self):
        self.update_data()

        pipe_found = self.pipe_results.heuristic_score > 0

        self.center = DownwardTarget(lambda self=self: (self.pipe_results.center_x, self.pipe_results.center_y),
                                     target=lambda self=self: get_camera_center(self.pipe_results),
                                     deadband=(20,20), px=0.003, py=0.003, dx=0.001, dy=0.001,
                                     valid=pipe_found)
        self.logi("Beginning to center on the pipe")

    def on_run(self):
        self.update_data()
        self.center()

        if self.center.finished:
            VelocityX(0)()
            VelocityY(0)()
            self.finish()

class align(Task):
    def on_first_run(self):
        self.align = PIDLoop(output_function=RelativeToCurrentHeading(), target=0, input_value=lambda: shm.pipe_results.angle.get(), negate=True, deadband=1)
        self.alignment_checker = ConsistencyCheck(19, 20)

        self.logi("Beginning to align to the pipe's heading")

    def on_run(self):
        self.align()

        if self.alignment_checker.check(self.align.finished):
            self.finish()

class mark(Task):
    def on_run(self, grp):
        cur = grp.get()
        cur.heading = math.radians(shm.pipe_results.angle.get())
        grp.set(cur)
        self.finish()

class turn(Task):
    def on_first_run(self):
        
        self.yaw = AbsoluteYaw(TORPEDO_OFFSET)
        self.yaw()
    
    def on_run(self):
        
        if self.yaw.finished:
            self.yaw.finish()
            self.finish()

        self.yaw()


class Pause(Task):
    def on_first_run(self, wait):
        self.start = time.time()

    def on_run(self, wait=4):
        if self.this_run_time - self.start > wait:
            self.finish()

class CheckBuoy(Task):
    def on_first_run(self):
        if BuoyFail is False:
            self.finish()
            return
        self.go = ToRedBuoy

    def on_run(self):
        print("Navigating to buoy")
        self.go()

class CheckWire(Task):
    def on_first_run(self):
        if WireFail is False:
            self.finish()
            return
        self.go = ToPortal

    def on_run(self):
        print("Navigating to wire")
        self.go()

class Forward(Task):
    def on_first_run(self):
        self.go = VelocityX()
        self.start = time.time()

    def on_run(self):
        self.go(.7)
        if self.this_run_time - self.start > 6:
            self.go(0)
            self.go.finish()
            self.finish()

class CheckPipe(Task):
    def on_first_run(self):
        self.c = center()
        self.a = align()
        self.state = 'c'
        self.start = time.time()
        if not (shm.pipe_results.heuristic_score.get() > 3000 and shm.pipe_results.rectangularity.get() > .75):
            self.finish()

    def on_run(self):
        if self.state is 'c':
            self.c()
            if self.this_run_time - self.start > 10:
                self.c.finish()
                self.start = time.time()
                self.state = 'a'
                return
        if self.state is 'a':
            self.a()
            if self.this_run_time - self.start > 6:
                self.a.finish()
                self.finish()



HeadingCenter = lambda: None
PipeCenter = center()
search = Search()
lineup = align()
test1 = Sequential(PipeCenter, lineup, VelocityX(.3))

check = CheckBuoy()

PipeNV = Sequential(Search(), center(), align(), VelocityX(.25))
#PipeQ = Sequential(VisionModule("Recovery"), Pause(3), Search(20), center(), align())
#Pipe = lambda begin, end: Sequential(VisionModule(begin), VisionModule(end, stop=True), Search(), center(), align(), Finite(Depth(2.2)))
#Vision = lambda begin, end: Sequential(VisionModule(begin), VisionModule(end, stop=True))
#BPipe = lambda begin, end: Sequential(VisionModule(begin), VisionModule(end, stop=True), center(), align(), Finite(Depth(2.4)))
#PipeNoSearch = lambda begin, end: Sequential(VisionModule(begin), VisionModule(end, stop=True), Pause(2), SurgeToPipe(), center(), align())
#FirstPipe = Sequential(VisionModule("redbuoy"), center(), align())
#
#
#tobuoy = Sequential(FirstPipe, Forward(), MasterConcurrent(ToBuoy(), VelocityX(.6)), VelocityX(0))#, CheckBuoy())
#towire = Sequential(Vision(begin="portal", end="redbuoy"), CheckPipe(), Finite(Depth(2.2)), MasterConcurrent(ToWire(), VelocityX(.35)), VelocityX(0))#, CheckWire())
#totorpedoes = Sequential(PipeNoSearch(begin="torpedoes", end="portal"), MasterConcurrent(ToTorpedoes(), VelocityX(.25)), VelocityX(0))
#tobins = Sequential(turn(), VisionModule("bins"), VisionModule("torpedoes", stop=True), MasterConcurrent(ToBins(), VelocityX(.3)), VelocityX(0))
#
#
#start = Sequential(Finite(Depth(.35)), MasterConcurrent(ToBuoy(timeout=70), VelocityX(.3)), VelocityX(0))

search_task = lambda: SearchFor(SwaySearch(1, 1),
                                lambda: shm.pipe_results.heuristic_score.get() > 0)
one_pipe = lambda grp: Sequential(Depth(PIPE_SEARCH_DEPTH),
                              search_task(), center(), align(), mark(grp),
                              Depth(PIPE_FOLLOW_DEPTH))

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

class OptimizablePipe(Task):
  def possibleModes(self):
    if self.finished:
      return []

    return [
      Mode(name = 'AlignAndGrab', expectedPoints = 100, expectedTime = 30)
      ]

  def desiredModules(self):
    return [shm.vision_modules.Pipes]

  def on_first_run(self, mode, grp):
    self.subtask = Timeout(30.0, one_pipe(grp))

  def on_run(self, mode, grp):
    self.subtask()
    if self.subtask.finished:
      self.finish()
