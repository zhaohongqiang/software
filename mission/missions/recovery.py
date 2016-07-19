import math
import random
import time
from collections import namedtuple, deque
import itertools
import numpy as np

import shm
from shm import recovery_state as world
from shm.watchers import watcher
from mission.framework.task import Task
from mission.framework.targeting import DownwardTarget, PIDLoop
from mission.framework.combinators import Sequential, Concurrent, MasterConcurrent
from mission.framework.movement import Heading, Depth, RelativeToCurrentHeading, RelativeToCurrentDepth, RelativeToInitialDepth, Roll
from mission.framework.timing import Timer
from mission.framework.primitive import Zero, FunctionTask, NoOp, Log
from mission.framework.helpers import get_camera, ConsistencyCheck
from mission.framework.actuators import SetActuators
from mission.framework.position import MoveXY, MoveXYRough, MoveX, MoveY, GoToPosition
from mission.framework.search import SpiralSearch
from mission.framework.track import Tracker, ConsistentObject
from auv_python_helpers.angles import heading_sub_degrees
import aslam

"""
    ____                                          ___   ____ ________
   / __ \___  _________ _   _____  _______  __   |__ \ / __ <  / ___/
  / /_/ / _ \/ ___/ __ \ | / / _ \/ ___/ / / /   __/ // / / / / __ \
 / _, _/  __/ /__/ /_/ / |/ /  __/ /  / /_/ /   / __// /_/ / / /_/ /
/_/ |_|\___/\___/\____/|___/\___/_/   \__, /   /____/\____/_/\____/
                                     /____/
"""

# TODO: rearrange stuff in file to make more sense
# TODO: consider logging successes and failures of same task in same task

class Vision(Task):
    STACK_FIELDS = ['visible', 'red', 'x', 'y', 'area', 'aspect_ratio', 'angle']
    MARK_FIELDS = ['visible', 'x', 'y', 'area']
    REGION_FIELDS = ['visible', 'x', 'y', 'area']
    Stack = namedtuple('Stack', STACK_FIELDS)
    Mark = namedtuple('Mark', MARK_FIELDS)
    Region = namedtuple('Region', REGION_FIELDS)
    COLORS = ['red', 'green']
    TRACKING_REJECT_WIDTH_RATIO = 0.15

    def on_first_run(self, *args, **kwargs):
        self.watcher = watcher()
        self.watcher.watch(shm.recovery_vision)
        self.pull_shm()

        self.stacks = [None] * 4
        tracker = lambda: Tracker(self.cam_width * Vision.TRACKING_REJECT_WIDTH_RATIO)
        self.red_stack_tracker = tracker()
        self.green_stack_tracker = tracker()
        self.mark_mappings = {color: ConsistentObject() for color in Vision.COLORS}
        self.region_mappings = {color: ConsistentObject() for color in Vision.COLORS}

        self.pull()

    def on_run(self, *args, **kwargs):
        if self.watcher.has_changed():
            self.pull()

    def pull_shm(self):
        self.shm = shm.recovery_vision.get()

        cam = get_camera(self.shm)
        self.cam_width, self.cam_height = cam['width'], cam['height']
        self.cam_center = (self.cam_width / 2, self.cam_height / 2)

    def pull(self):
        self.pull_shm()
        self.pull_stacks()
        self.pull_marks()
        self.pull_regions()

    def pull_stacks(self):
        red_stacks, green_stacks = [], []
        for i in range(4):
            vals = {}
            for field in Vision.STACK_FIELDS:
                vals[field] = getattr(self.shm, 'stack_{}_{}'.format(i+1, field))
            stack = Vision.Stack(**vals)
            if stack.visible:
                if stack.red:
                    red_stacks.append(stack)
                else:
                    green_stacks.append(stack)
        pad_list = lambda x: x + ([None] * (2 - len(x)))
        red_stacks = pad_list(red_stacks)
        green_stacks = pad_list(green_stacks)

        new_red_stacks = self.red_stack_tracker.track(*red_stacks)
        new_green_stacks = self.green_stack_tracker.track(*green_stacks)
        for i, s in enumerate(new_red_stacks + new_green_stacks):
            if s is not None and self.stacks[i] is None:
                self.logv('Started tracking {} stack at index {}'.format(
                    'red' if s.red else 'green', i))
            elif s is None and self.stacks[i] is not None:
                self.logv('Stopped tracking {} stack at index {}'.format(
                    'red' if self.stacks[i].red else 'green', i))
            self.stacks[i] = s

        self.debug_locations(self.stacks, 0)

    def pull_marks(self):
        self.marks = []
        for color in Vision.COLORS:
            vals = {}
            for field in Vision.MARK_FIELDS:
                vals[field] = getattr(self.shm, '{}_mark_{}'.format(color, field))
            mark = Vision.Mark(**vals)
            self.marks.append(self.mark_mappings[color].map(mark))

        self.debug_locations(self.marks, len(self.stacks))

    def pull_regions(self):
        self.regions = []
        for color in Vision.COLORS:
            vals = {}
            for field in Vision.REGION_FIELDS:
                vals[field] = getattr(self.shm, '{}_region_{}'.format(color, field))
            region = Vision.Region(**vals)
            self.regions.append(self.region_mappings[color].map(region))

        self.debug_locations(self.regions, len(self.stacks) + len(self.marks))

    def debug_locations(self, objects, index_offset, coord_offset=(0, 0)):
        vision_debug = shm.vision_debug.get()
        for i, obj in enumerate(objects):
            if obj is not None and obj.visible:
                setattr(vision_debug, 'x{}'.format(i + index_offset), int(obj.x + coord_offset[0]))
                setattr(vision_debug, 'y{}'.format(i + index_offset), int(obj.y + coord_offset[1]))
                setattr(vision_debug, 'text{}'.format(i + index_offset), bytes(str(i + index_offset), encoding='utf-8'))
            else:
                setattr(vision_debug, 'text{}'.format(i + index_offset), b'')

        shm.vision_debug.set(vision_debug)

class Timeout(Task):
    """
    Try doing a task for a certain amount of time

    We are successful if the task completes in time and is successful.
    """
    def on_first_run(self, task, time, *args, **kwargs):
        self.success = False
        self.task = task
        self.timer = Timer(time)
        self.timed_out = False

    def on_run(self, *args, **kwargs):
        self.task()
        self.timer()
        if self.task.finished:
            if hasattr(self.task, 'success'):
                self.success = self.task.success
            else:
                self.success = True
            self.finish()
        elif self.timer.finished:
            self.timed_out = True
            self.finish()

class SuccessOverride(Task):
    def on_run(self, task, override=True, *args, **kwargs):
        task()
        if task.finished:
            self.success = override
            self.finish()

Success = lambda task: SuccessOverride(task, override=True)
Failure = lambda task: SuccessOverride(task, override=False)

class Retry(Task):
    """
    Keep attempting a task until it succeeds, or until it has been attempted a given
    number of times.
    """
    def on_first_run(self, task_func, attempts, *args, **kwargs):
        self.success = False
        self.task = task_func()
        self.task_name = self.task.__class__.__name__
        self.attempt = 0
        self.log_attempt = True

    def on_run(self, task_func, attempts, *args, **kwargs):
        if self.attempt < attempts:
            if self.log_attempt:
                self.logi('Attempt {} of {} at {}'.format(self.attempt + 1, attempts, self.task_name))
                self.log_attempt = False

            self.task()
            if self.task.finished:
                if self.task.success:
                    self.logi('{} succeeded on attempt {}!'.format(self.task_name, self.attempt + 1))
                    self.success = True
                    self.finish()
                else:
                    # If at first we don't succeed, umm, try, try again
                    self.attempt += 1
                    self.log_attempt = True
                    self.task = task_func()
        else:
            self.loge('Failed {} after {} attempts'.format(self.task_name, attempts))
            self.finish()

class SequentialSuccess(Task):
    def on_first_run(self, *args, **kwargs):
        self.success = False

    def on_run(self, *tasks, subtasks=(), finite=True, **kwargs):
        subtasks_iterable = itertools.chain(tasks, subtasks)

        for task in subtasks_iterable:
            if finite and task.has_ever_finished:
                continue
            task()
            if finite and task.has_ever_finished:
                if hasattr(task, 'success') and not task.success:
                    self.finish()
                    break
            if not task.finished:
                break
        else:
            self.success = True
            self.finish()

class Recovery(Task):
    def on_first_run(self, vision, reset_state=True, *args, **kwargs):
        if reset_state:
            w = world.get()
            w.stacks_on_tower = 4
            w.first_hstack_removed = 0
            w.grabber_stack_present = 0
            w.stacks_on_table = 0
            world.set(w)

        get_stack = lambda: Retry(lambda: GetStack(vision), 4)

        self.task = SequentialSuccess(subtasks=[
            SequentialSuccess(get_stack(), Surface(), PlaceStack(vision)) for i in range(4)])

    def on_run(self, vision, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()
            if self.task.success:
                self.logi('Success!')
            else:
                self.loge('Failure :(')

class GetStack(Task):
    """
    Pick up one stack from the tower, including moving to the tower and grabbing a stack
    """
    def on_first_run(self, vision, *args, **kwargs):
        self.success = False
        self.choose_and_grab = ChooseAndGrab(vision)
        self.task = SequentialSuccess(
            MoveAboveTower(vision),
            self.choose_and_grab,
        )

    def on_run(self, vision, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.success = self.task.success
            self.finish()

    def on_finish(self, *args, **kwargs):
        if self.success:
            self.logi('Successfully grabbed stack!')
            world.stacks_on_tower.set(world.stacks_on_tower.get() - 1)
            choice = self.choose_and_grab.grab_choice
            world.grabber_stack_present.set(1)
            world.grabber_stack_red.set(choice.red)
            if choice.heading is not None:
                world.second_hstack_heading.set((choice.heading + 180) % 360)
        else:
            self.loge('Failed to grab stack')

class ChooseAndGrab(Task):
    """
    Choose a stack and try to grab it

    Begin: above tower looking at stacks
    """
    def on_first_run(self, vision, *args, **kwargs):
        self.success = False
        self.grab_choice = choose_next_stack(vision)
        if self.grab_choice.error_msg is not None:
            self.loge(self.grab_choice.error_msg)
            self.finish()
            return
        self.logi('Chose to grab {} {} stack at index {}'.format(
            'red' if self.grab_choice.red else 'green',
            'vertical' if self.grab_choice.vertical else 'horizontal',
            self.grab_choice.index,
        ))

        grab_task = None
        if self.grab_choice.vertical:
            grab_task = GrabVerticalStack(vision, self.grab_choice.index)
        else:
            grab_task = GrabHorizontalStack(
            vision, self.grab_choice.index, self.grab_choice.heading)

        initial_stacks = sum(stack is not None and stack.visible for stack in vision.stacks)
        self.task = SequentialSuccess(
            GrabAndRestore(vision, grab_task),
            VerifyGrab(vision, initial_stacks),
        )

    def on_run(self, vision, *args, **kwargs):
        if self.has_ever_finished:
            return
        if not self.task.finished:
            self.task()
        else:
            self.success = self.task.success
            self.finish()

class GrabAndRestore(Task):
    """
    Attempt to grab a stack and restore our initial position after
    """
    def on_first_run(self, vision, grab_task, *args, **kwargs):
        self.success = False

        north, east, depth = aslam.sub.position()
        self.task = Sequential(
            grab_task,

            Log('Moving to position before grab attempt'),
            Depth(depth),
            GoToPosition(north, east, optimize=False),
        )

    def on_run(self, vision, grab_task, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.success = grab_task.success
            self.finish()

class VerifyGrab(Task):
    """
    Check if the number of stacks on the tower is less than what we started
    with
    """
    def on_first_run(self, vision, original_stacks, *args, **kwargs):
        if original_stacks > 1:
            self.downward_target = SequentialSuccess(
                DownwardTargetObjects(vision, lambda: vision.stacks),
                Zero(),
            )
        else:
            self.downward_target = Success(Log('Not targeting stacks to verify grab, none should remain'))

    def on_run(self, vision, original_stacks, *args, **kwargs):
        if not self.downward_target.finished:
            self.downward_target()
        else:
            current_stacks = sum(stack is not None for stack in vision.stacks)
            self.success = self.downward_target.success and current_stacks < original_stacks
            if not self.success:
                self.loge('Grab failed, started with {} stacks but {} remain'.format(
                    original_stacks, current_stacks))

            # self.success = random.random() < 0.5
            self.success = True # TODO fix
            self.finish()

# Pause initially to give object-identifying tasks time to check current state
Search = lambda: Sequential(Timer(0.5), SpiralSearch(
    relative_depth_range=0,
    optimize_heading=True,
    meters_per_revolution=1,
    min_spin_radius=1,
))

class MoveAboveTower(Task):
    """
    Move to above the tower as quickly as possible given the current known information
    """
    # TODO: support pinger tracking
    min_search_stacks = 2
    default_altitude = 2.8

    def on_first_run(self, vision, *args, **kwargs):
        self.success = False

        stacks_on_tower = world.stacks_on_tower.get()
        go_to_tower = None
        if stacks_on_tower < 4: # Only rely on tower position after we've picked up a stack
            north = world.tower_north.get()
            east = world.tower_east.get()
            depth = world.tower_depth.get()

            go_to_tower = Sequential(
                Log('Returning to tower position ({}, {}, {})'.format(north, east, depth)),
                Depth(depth), # Move to tower depth early so we don't crash into tower
                GoToPosition(north, east, optimize=True),
            )

        else:
            go_to_tower = Sequential(
                Log('Going to default tower altitude of {}'.format(self.default_altitude)),
                Altitude(self.default_altitude),
            )

        search_tower = None
        if stacks_on_tower > 0:
            search_stacks = 2 if stacks_on_tower >= 2 else 1
            search_tower = Sequential(
                Log('Searching for tower'),
                Altitude(self.default_altitude),
                MasterConcurrent(
                    IdentifyObjects(lambda: vision.stacks, min_objects=search_stacks),
                    Search(),
                ),
            )
        else:
            search_tower = Log('Not searching for tower, no stacks on tower')

        center_tower = None
        if stacks_on_tower > 0:
            center_tower = SequentialSuccess(
                Log('Centering tower'),
                DownwardTargetObjects(vision, lambda: vision.stacks, fast=stacks_on_tower < 4),
                Zero(),
            )
        else:
            center_tower = Log('No stacks on tower, not centering')

        self.task = SequentialSuccess(go_to_tower, search_tower, center_tower)

    def on_run(self, vision, *args, **kwargs):
        if not self.task.finished:
            self.task()

        else:
            if self.task.success:
                self.success = True
                north, east, depth = aslam.sub.position()
                world.tower_north.set(north)
                world.tower_east.set(east)
                world.tower_depth.set(depth)
            else:
                self.loge('Failed to move above tower')

            self.finish()

class IdentifyObjects(Task):
    """
    Finish when some objects we are looking for are in view
    """
    def on_run(self, objects_func, min_objects=1, *args, **kwargs):
        n = sum(obj is not None and obj.visible for obj in objects_func())
        if n >= min_objects:
            self.finish()

class DownwardTargetObjects(Task):
    """
    Downward target the center of all provided objects

    Begin: at least one object in view
    End: center of all objects in center of camera
    """
    fast_p = 0.002
    fast_deadband = (40, 40)
    precise_p = 0.001
    precise_deadband = (10, 10)

    def centroid(self, objects):
        total_objects = 0
        center_x, center_y = 0, 0
        for obj in objects:
            if obj is not None and obj.visible:
                center_x += obj.x
                center_y += obj.y
                total_objects += 1

        center_x /= total_objects
        center_y /= total_objects
        return (center_x, center_y)

    def on_first_run(self, vision, objects_func, fast=True, *args, **kwargs):
        self.success = False
        self.task = DownwardTarget(
            point=lambda: self.centroid(self.objects),
            target=vision.cam_center,
            deadband=self.fast_deadband if fast else self.precise_deadband,
            px=self.fast_p if fast else self.precise_p,
            py=self.fast_p if fast else self.precise_p,
        )

    def on_run(self, vision, objects_func, *args, **kwargs):
        self.objects = objects_func()
        num_objects = sum(obj is not None and obj.visible for obj in self.objects)
        if num_objects == 0:
            self.loge("Can't see any objects, targeting aborted")
            self.finish()
            return

        self.task()
        if self.task.finished:
            self.success = True
            self.finish()

GrabChoice = namedtuple('GrabChoice', ['error_msg', 'index', 'red', 'vertical', 'heading'])

def choose_next_stack(vision):
    """
    Decide which stack to grab on the tower

    Current stack schedule: Pick up all vertical stacks, then all horizontal stacks.
    Needs to above the tower with all stacks in vision.
    """
    HORIZONTAL_ASPECT_RATIO = 2

    failedChoice = lambda info: GrabChoice(info, None, None, None, None)

    vstack_indices, hstack_indices = [], []
    for i, stack in enumerate(vision.stacks):
        if stack is not None:
            if stack.aspect_ratio >= HORIZONTAL_ASPECT_RATIO:
                hstack_indices.append(i)
            else:
                vstack_indices.append(i)
    vstack_indices = []

    target_indices = None
    if len(vstack_indices) > 0:
        target_indices = vstack_indices
    else:
        target_indices = hstack_indices
    if len(target_indices) == 0:
        return failedChoice('No stacks found to grab')
    if len(target_indices) > 2:
        return failedChoice('More than 2 {} stacks found, cannot grab'.format(
            'vertical' if target_indices is vstack_indices else 'horizontal'))

    target_index = target_indices[int(random.random() * len(target_indices))]
    target_stack = vision.stacks[target_index]

    heading = None
    if target_index in hstack_indices:
        if len(hstack_indices) == 2:
            target_pos = np.array([target_stack.x, target_stack.y])
            other_index = None
            if target_index == hstack_indices[0]:
                other_index = hstack_indices[1]
            else:
                other_index = hstack_indices[0]
            other_stack = vision.stacks[other_index]

            avg_pos = np.array([
                (target_stack.x + other_stack.x) / 2,
                (target_stack.y + other_stack.y) / 2,
            ])
            stack_vec = target_pos - avg_pos

            heading = None
            heading = math.degrees(math.atan2(stack_vec[0], -stack_vec[1]))
            heading += shm.kalman.heading.get() # Now a global heading
            heading -= 90 # Align to the stack on the port side
            heading %= 360

        elif len(hstack_indices) == 1:
            if world.first_hstack_removed.get():
                heading = world.second_hstack_heading.get()
            else:
                return failedChoice("Can't characterize horizontal stack in vision, need info on other horizontal stack")

    return GrabChoice(
        error_msg=None,
        index=target_index,
        red=target_stack.red,
        vertical=len(vstack_indices) > 0,
        heading=heading,
    )

def ExtendAmlan():
    return SetActuators(['piston_extend'], ['piston_retract'])

def RetractAmlan():
    return SetActuators(['piston_retract'], ['piston_extend'])

class GradualHeading(Task):
    def on_first_run(self, desire, *args, **kwargs):
        subtasks = []
        increment = 20
        deadband = 25

        current = shm.kalman.heading.get()
        while abs(heading_sub_degrees(current, desire)) % 360 > deadband:
            current -= math.copysign(increment, heading_sub_degrees(current, desire))
            subtasks.append(Heading(current))

        self.task = Sequential(subtasks=subtasks)

    def on_run(self, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

class GrabStack(Task):
    """
    Grabs the stack at the given index at the given approximate heading
    """
    ALIGN_STACK_P = 0.5
    ALIGN_STACK_DEADBAND = 3
    DOWNWARD_TARGET_DEADBAND = (30, 30)

    def on_first_run(self, vision, stack_i, timeout, dive_task, offset_task=None, approx_heading=None, snap_move=None, slide_move=None, *args, **kwargs):
        self.success = False
        self.must_see_stack = True

        downward_target = lambda fast=True: DownwardTargetObjects(vision, lambda: [vision.stacks[stack_i]], fast=fast)
        precise_align = PIDLoop(
            lambda: vision.stacks[stack_i].angle,
            RelativeToCurrentHeading(),
            target=90,
            negate=True,
            p=GrabStack.ALIGN_STACK_P,
            deadband=GrabStack.ALIGN_STACK_DEADBAND,
        )

        def ignore_stack():
            self.must_see_stack = False

        self.task = Timeout(Sequential(
            Log('Aligning to stack'),
            downward_target(),
            Zero(),

            Concurrent(
                Sequential(
                    Log('Roughly aligning to stack angle {}'.format(approx_heading)),
                    GradualHeading(approx_heading),
                    Log('Precisely aligning to stack angle'),
                    precise_align,
                ),
                downward_target(),
                finite=False,
            ) if approx_heading is not None else NoOp(),

            Log('Going down closer to stack'),
            Concurrent(dive_task, downward_target(fast=False), finite=False),
            Zero(),

            FunctionTask(ignore_stack),
            Sequential(
                Log('Applying offset'),
                offset_task,
            ) if offset_task is not None else NoOp(),

            Log('Moving to stack'),
            snap_move if snap_move is not None else NoOp(),

            Log('Extending Amlan'),
            ExtendAmlan(),
            Timer(2),

            Log('Sliding stack'),
            slide_move if slide_move is not None else NoOp(),

            Log('Hopefully grabbed stack'),
        ), timeout)

    def on_run(self, vision, stack_i, timeout, *args, **kwargs):
        if self.must_see_stack and vision.stacks[stack_i] is None:
            self.loge('Lost stack')
            self.finish()
            return

        if not self.task.finished:
            self.task()
        else:
            self.finish()
            if self.task.success:
                self.success = True
            else:
                self.loge('Failed after {} seconds'.format(timeout))

    def on_finish(self, *args, **kwargs):
        if not self.success:
            RetractAmlan()()

MoveGrabberToCamera = lambda: MoveXY((-0.1, 0.17), deadband=0.008)

def GrabVerticalStack(vision, stack_i):
    return GrabStack(
        vision, stack_i,
        timeout=60,
        dive_task=Altitude(1.70),
        offset_task=MoveGrabberToCamera(),
        # snap_move=AltitudeUntilStop(1),
        snap_move=AltitudeUntilStop(1.5), # TODO fix
    )

def GrabHorizontalStack(vision, stack_i, approx_heading):
    return GrabStack(
        vision, stack_i,
        timeout=90,
        dive_task=Altitude(1.54),
        offset_task=MoveGrabberToCamera(),
        approx_heading=approx_heading,
        # snap_move=AltitudeUntilStop(1),
        snap_move=AltitudeUntilStop(1.5), # TODO fix
        slide_move=MoveXY((0, 0.3)),
    )

class Altitude(Task):
    def on_first_run(self, altitude, p=0.5, d=0.1, deadband=0.05, *args, **kwargs):
        self.task = PIDLoop(
            shm.dvl.savg_altitude.get,
            RelativeToCurrentDepth(),
            target=altitude,
            p=p,
            d=d,
            negate=True,
            deadband=deadband,
        )

    def on_run(self, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

class AltitudeUntilStop(Task):
    """
    Attempt to move to the target altitude until the sub gets stuck
    """
    def on_first_run(self, altitude, *args, **kwargs):
        self.min_speed = 0.008
        self.min_delta = 0.1
        self.deque_size = 20

        self.success = False
        self.altitude_task = Altitude(altitude, p=0.3)
        self.stop_cons_check = ConsistencyCheck(3, 3)
        self.readings = deque()
        self.initial_altitude = shm.dvl.savg_altitude.get()
        self.last_altitude = self.initial_altitude

    def on_run(self, altitude, *args, **kwargs):
        if not self.altitude_task.has_ever_finished:
            self.altitude_task()
            current_altitude = shm.dvl.savg_altitude.get()
            self.readings.append((current_altitude, time.time()))
            if len(self.readings) > self.deque_size:
                self.readings.popleft()

            if abs(current_altitude - self.initial_altitude) >= self.min_delta and \
                    len(self.readings) >= self.deque_size:
                delta_altitude = self.readings[-1][0] - self.readings[0][0]
                delta_time = self.readings[-1][1] - self.readings[0][1]
                speed = abs(delta_altitude / delta_time)

                if self.stop_cons_check.check(speed < self.min_speed):
                    self.logi('Stopped changing altitude, finishing')
                    self.success = True
                    self.finish()
        else:
            self.loge('Bounding altitude reached')
            self.finish()

class Wiggle(Task):
    def on_first_run(self, *args, **kwargs):
        self.task = Sequential(
            MoveX(-0.05, deadband=0.03),
            MoveX(0.1, deadband=0.04),
            MoveX(-0.1, deadband=0.04),
            MoveX(0.1, deadband=0.04),
            MoveX(-0.1, deadband=0.04),
            MoveX(0.1, deadband=0.04),
            MoveX(-0.1, deadband=0.04),
            MoveX(0.05, deadband=0.03),
        )

    def on_run(self, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

class Surface(Task):
    """
    Breaches the surface of the water inside the octogon for a fixed time

    Begin: Sub centered and zeroed directly over tower
    End: Sub slightly below surface
    """
    # Don't surface to a negative depth, it pushes bubbles under the sub onto the camera and dvl
    pre_surface_depth = 0.5
    surface_depth = 0
    surface_time = 3

    def on_first_run(self, *args, **kwargs):
        original_depth = shm.kalman.depth.get()

        self.task = Sequential(
            Log('Rising to just below surface'),
            Depth(self.pre_surface_depth),

            Log('Surfacing!'),
            MasterConcurrent(Timer(self.surface_time), Depth(self.surface_depth)),

            Log('Falling back below surface'),
            Depth(original_depth),
        )

    def on_run(self, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

class PlaceStack(Task):
    """
    Locate the table and place the stack we're holding on it

    Begin: holding a stack
    End: above the table after placing stack
    """
    move_above_table_max_retries = 4

    def on_first_run(self, vision, *args, **kwargs):
        self.task = SequentialSuccess(
            Log('Moving above table'),
            Retry(lambda: MoveAboveTable(vision), self.move_above_table_max_retries),

            Log('Dropping stack'),
            DropStack(vision),
        )

    def on_run(self, vision, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

    def on_finish(self, *args, **kwargs):
        self.success = self.task.success
        if self.success:
            world.stacks_on_table.set(world.stacks_on_table.get() + 1)
        else:
            self.loge('Failed to place stack')

class MoveAboveTable(Task):
    """
    Move to above the table as quickly as possible given the current known information

    Start: Anywhere
    End: Zeroed centered on the table
    """
    def on_first_run(self, vision, *args, **kwargs):
        self.success = False

        if world.stacks_on_table.get() > 0:
            # Marks could be distorted, can't center on them to center table
            north = world.table_north.get()
            east = world.table_east.get()
            depth = world.table_depth.get()

            self.task = SequentialSuccess(
                Log('Returning to table position ({}, {}, {})'.format(north, east, depth)),
                GoToPosition(north, east, optimize=True),
                Depth(depth), # Move to table depth late so we don't crash into tower

                Log('Centering colored table regions'),
                DownwardTargetObjects(vision, lambda: vision.regions),
                Zero(),
            )

        else:
            # No stacks on table, so we need to find it and record its
            # position
            def record_position():
                north, east, depth = aslam.sub.position()
                world.table_north.set(north)
                world.table_east.set(east)
                world.table_depth.set(depth)

            self.task = SequentialSuccess(
                Log('Searching for table marks'),
                MasterConcurrent(IdentifyObjects(lambda: vision.marks), Search()),

                Log('Centering table marks'),
                DownwardTargetObjects(vision, lambda: vision.marks),

                Log('Centering colored table regions'),
                DownwardTargetObjects(vision, lambda: vision.regions, fast=False),
                Zero(),

                Log('Recording table position'),
                FunctionTask(record_position),
            )

    def on_run(self, vision, *args, **kwargs):
        if not self.task.finished:
            self.task()

        else:
            if self.task.success:
                self.success = True

                # The best time to record position is if we've never placed a
                # stack
                if world.stacks_on_table.get() == 0:
                    north, east, depth = aslam.sub.position()
                    world.table_north.set(north)
                    world.table_east.set(east)
                    world.table_depth.set(depth)
            else:
                self.loge('Failed to move above table')

            self.finish()

class DropStack(Task):
    """
    Try dropping the stack we're holding on the given mark color once

    Start: target mark visible in downcam
    End: above the table with stack dropped, or not
    """
    precise_align_altitude = 1.25
    drop_altitude = 0.75
    retract_amlan_time = 0.5
    def on_first_run(self, vision, *args, **kwargs):
        self.success = False

        if not world.grabber_stack_present.get():
            self.loge('No stack present in grabber, cannot drop')
            self.finish()
        red = world.grabber_stack_red.get()

        def empty_grabber():
            world.grabber_stack_present.set(0)

        target_region = lambda fast=True: DownwardTargetObjects(
            vision, lambda: [vision.regions[0 if red else 1]], fast=fast)

        self.task = SequentialSuccess(
            Log('Targeting {} region'.format('red' if red else 'green')),
            target_region(),

            Log('Going down to target region more accurately'),
            Concurrent(
                Altitude(self.precise_align_altitude),
                target_region(fast=False),
                finite=False,
            ),
            Zero(),

            Log('Aligning region with grabber'),
            MoveGrabberToCamera(),

            Log('Going down to drop'),
            Altitude(self.drop_altitude),

            Log('Retracting Amlan'),
            RetractAmlan(),
            Timer(self.retract_amlan_time),
            FunctionTask(empty_grabber),
        )

    def on_run(self, vision, *args, **kwargs):
        if self.has_ever_finished:
            return

        if not self.task.finished:
            self.task()
        else:
            self.success = self.task.success
            self.finish()
            if not self.success:
                self.loge('Failed to drop stack')

def VisionTask(task_class, *args, **kwargs):
    vision = Vision()
    task = task_class(vision, *args, **kwargs)
    return MasterConcurrent(Sequential(Timer(1), task), vision)

def SimulatedTask(task):
    def update_altitude():
        shm.dvl.savg_altitude.set(3 - shm.kalman.depth.get())

    return MasterConcurrent(task, FunctionTask(update_altitude, finite=False))

recovery = lambda: VisionTask(Recovery)
recovery_noreset = lambda: VisionTask(Recovery, reset=False)
sim_recovery = lambda: SimulatedTask(recovery())
vision = lambda: Vision()
target_stacks = lambda: VisionTask(TargetStacks)
grab_next = lambda: VisionTask(GrabNextStack, ignore_indices=[0, 2])
grab_vstack = lambda: VisionTask(GrabVerticalStack, 1)
sim_grab_vstack = lambda: SimulatedTask(grab_vstack())
grab_hstack = lambda: VisionTask(GrabHorizontalStack, 3, 0)
altitude_until_stop = lambda: AltitudeUntilStop(1)
move = lambda: MoveXYRough((-1, 0.5))
go_to_position = lambda: GoToPosition(0, 0, optimize=True)
altitude = lambda: Altitude(2)

sim_move_above_tower = lambda: SimulatedTask(VisionTask(MoveAboveTower))
sim_get_stack = lambda: SimulatedTask(VisionTask(GetStack))
sequential_success = lambda: SequentialSuccess(Timeout(NoOp(finite=False), 1), Log('next'))
place_stack = lambda: VisionTask(PlaceStack)
sim_place_stack = lambda: SimulatedTask(VisionTask(PlaceStack))
surface = lambda: Surface()
