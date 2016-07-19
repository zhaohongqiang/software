import numpy as np
import shm
from shm import kalman
from shm.watchers import watcher
import aslam
from mission.framework.task import Task
from mission.framework.combinators import Concurrent, Sequential
from mission.framework.movement import Heading, Roll, Pitch, Depth, VelocityX, VelocityY, \
    RelativeToCurrentHeading, RelativeToCurrentDepth
from mission.framework.position import MoveX, MoveY
from mission.framework.targeting import PIDLoop
from mission.framework.timing import Timer
from mission.framework.helpers import get_camera_size, call_if_function
from mission.framework.primitive import Zero
from mission.opt_aux.aux import Mode

vision = None

class Vision(Task):
    """ Get vision results and call the main task
    """
    def on_first_run(self, main_task, *args, **kwargs):
        self.main_task = main_task
        self.watcher = watcher()
        self.watcher.watch(shm.navigate_results)
        self.pull()

    def on_run(self, main_task, *args, **kwargs):
        if self.watcher.has_changed():
            self.pull()

        if not self.main_task.has_ever_finished:
            self.main_task(main_task, *args, **kwargs)
        else:
            self.finish()

    def pull(self):
        global vision
        vision = shm.navigate_results.get()

class Style(Task):
    """Base class for all styles

    Start: facing center of gate
    Finish: facing away from center of gate
    """
    def on_first_run(self, *args, **kwargs):
        # `self.__class__.__name__` gets the name of the class from an instance
        self.logi('Starting')
        self.style_on_first_run(*args, **kwargs)

    def on_run(self, *args, **kwargs):
        self.style_on_run(*args, **kwargs)

    def on_finish(self, *args, **kwargs):
        self.style_on_finish(*args, **kwargs)
        Zero()()
        self.logi('Finished in {} seconds!'.format(
            self.this_run_time - self.first_run_time))

    """
    These should be overridden by child style classes
    """
    def style_on_first_run(self, *args, **kwargs):
        pass
    def style_on_run(self, *args, **kwargs):
        pass
    def style_on_finish(self, *args, **kwargs):
        pass

class StyleBasic(Style):
    """Simply moves forward
    """
    def style_on_first_run(self, distance=5, *args, **kwargs):
        self.movement = MoveX(distance)

    def style_on_run(self, *args, **kwargs):
        if not self.movement.has_ever_finished:
            self.movement()
        else:
            self.finish()

class StyleSideways(Style):
    """Heading changes 90 degrees starboard, so that sub is facing either right or left

    If `starboard` is False, then heading changes 90 degrees port
    """
    def style_on_first_run(self, starboard=True, *args, **kwargs):
        current_heading = kalman.heading.get()
        if starboard:
            change_heading = Heading(current_heading + 90, error=1)
            movement = MoveY(-5)
        else:
            change_heading = Heading(current_heading - 90, error=1)
            movement = MoveY(5)
        heading_restore = Heading(current_heading, error=1)
        self.style_sideways = Sequential(change_heading, movement, heading_restore)

    def style_on_run(self, *args, **kwargs):
        if not self.style_sideways.has_ever_finished:
            self.style_sideways()
        else:
            self.finish()

class StyleUpsideDown(Style):
    """Roll changes 180 degrees, so that sub is upside down
    """
    def style_on_first_run(self, *args, **kwargs):
        change_roll = Roll(180, error=1)
        movement = MoveX(5)
        restore_roll = Roll(0, error=1)
        self.style_upside_down = Sequential(change_roll, movement, restore_roll)

    def style_on_run(self, *args, **kwargs):
        if not self.style_upside_down.has_ever_finished:
            self.style_upside_down()
        else:
            self.finish()

class StylePitched(Style):
    """
    Pitch changes 75 degrees, so that sub is facing either down or up
    The reason for 75 degrees is so that th sub does not rapidly twist back
    and forth, in an attempt to maintain a stable heading

    If `up` is False, then sub pitches downwards
    """
    def style_on_first_run(self, up=True, *args, **kwargs):
        if up:
            change_pitch = Pitch(75, error=1)
        else:
            change_pitch = Pitch(-75, error=1)
        movement = MoveX(5)
        restore_pitch = Pitch(0, error=1)
        self.style_pitched = Sequential(change_pitch, movement, restore_pitch)

    def style_on_run(self, *args, **kwargs):
        if not self.style_pitched.has_ever_finished:
            self.style_pitched()
        else:
            self.finish()

class StyleLoop(Style):
    """Does a loop around the center bar of the channel

    Goes forward and under, backwards and over, then forwards and over
    """
    def style_on_first_run(self, *args, **kwargs):
        move_distance = 5 # meters
        depth_offset = 1 # offset to go up or down

        def generate_curve(distance, depth_offset, depth, iterations):
            #TODO: Make this curve more 'curvy'
            movement = []
            dist_tick = distance / iterations
            current_depth = depth
            depth_tick = depth_offset / (iterations - 1)
            for t in range(iterations):
                movement.append(Concurrent(MoveX(dist_tick), Depth(current_depth, error=.1)))
                current_depth += depth_tick
            return Sequential(subtasks=movement)

        current_depth = kalman.depth.get()
        forward_and_down = generate_curve(move_distance / 2, depth_offset, current_depth, 3)
        forward_and_up = generate_curve(move_distance / 2, -depth_offset, current_depth + depth_offset, 3)
        backward_and_up = generate_curve(-move_distance / 2, -depth_offset, current_depth, 3)
        backward_and_down = generate_curve(-move_distance / 2, depth_offset, current_depth - depth_offset, 3)
        forward = Sequential(generate_curve(move_distance / 2, -depth_offset, current_depth, 3),
                             generate_curve(move_distance / 2, depth_offset, current_depth - depth_offset, 3))
        self.style_loop = Sequential(forward_and_down, forward_and_up, backward_and_up,
                                     backward_and_down, forward)

    def style_on_run(self, *args, **kwargs):
        if not self.style_loop.has_ever_finished:
            self.style_loop()
        else:
            self.finish()

class StyleSuperSpin(Style):
    def style_on_first_run(self, clockwise=True, steps=5, spins=2, *args, **kwargs):
        initial_heading = shm.kalman.heading.get()
        subspins = []
        for spin in range(spins):
            for step in range(steps):
                angle = 360 / steps * (step+1)
                if not clockwise: angle = 360 - angle
                subspins.append(Heading(initial_heading + angle, error=10))

        self.movement = Sequential(
            MoveY(0.75, deadband=0.1),
            MoveX(2, deadband=0.2),
            Concurrent(
                MoveX(3, deadband=0.2),
                Sequential(subtasks=subspins),
            )
        )

    def style_on_run(self, *args, **kwargs):
        if not self.movement.has_ever_finished:
            self.movement()
        else:
            self.finish()

DEFAULT_DEADBAND = 0.03

class AlignHeading(Task):
    def on_first_run(self, *args, **kwargs):
        self.pid = PIDLoop(
            input_value=self.x_ratio,
            output_function=RelativeToCurrentHeading(),
            target=0.5,
            deadband=DEFAULT_DEADBAND/3,
            p=40,
            d=20,
            negate=True,
        )

    def on_run(self, *args, **kwargs):
        # Try to center on entire gate first
        if vision.bottom_prob or (vision.left_prob and vision.right_prob):
            self.pid()

        elif vision.left_prob: # Otherwise try to find other vertical bar
            RelativeToCurrentHeading(1)()
        elif vision.right_prob:
            RelativeToCurrentHeading(-1)()

        if self.pid.finished:
            self.finish()

    def x_ratio(self):
        """ Precondition: we can see the bottom bar, or the left and right bars
        """
        CAM_WIDTH, CAM_HEIGHT = get_camera_size(vision)

        if vision.bottom_prob:
            avg_x = (vision.bottom_x1 + vision.bottom_x2) / 2
            return avg_x / CAM_WIDTH
        else: # We see left and right bars
            left_x_avg = (vision.left_x1 + vision.left_x2) / 2
            right_x_avg = (vision.right_x1 + vision.right_x2) / 2
            avg = (left_x_avg + right_x_avg) / 2
            return avg / CAM_WIDTH

class AlignDepth(Task):
    def on_first_run(self, *args, **kwargs):
        self.pid = PIDLoop(
            input_value=self.y_ratio,
            output_function=RelativeToCurrentDepth(),
            target=0.75,
            deadband=DEFAULT_DEADBAND,
            p=2,
            negate=True,
        )

    def on_run(self, *args, **kwargs):
        if vision.bottom_prob or vision.left_prob or vision.right_prob:
            self.pid()

        if self.pid.finished:
            self.finish()

    def y_ratio(self):
        """ Precondition: we can see at least one of the bars
        """
        CAM_WIDTH, CAM_HEIGHT = get_camera_size(vision)

        if vision.bottom_prob:
            avg_y = (vision.bottom_y1 + vision.bottom_y2) / 2
            return avg_y / CAM_HEIGHT
        elif vision.left_prob or vision.right_prob:
            return 1.1

class AlignFore(Task):
    def on_first_run(self, *args, **kwargs):
        self.pid = PIDLoop(
            input_value=self.width_ratio,
            output_function=VelocityX(),
            target=0.55,
            deadband=DEFAULT_DEADBAND,
            p=2,
        )

        # Maximum ratio of camera width bottom bar can be away from camera
        # edge
        self.EDGE_PROXIMITY = 0.05

        # Speed to back away from the bottom bar at when too close
        self.BACKUP_SPEED = 0.15

        # Speed to approach the gate at when not fully visible
        self.APPROACH_SPEED = 1

        # Min width of gate to begin carefully approaching
        self.MIN_WIDTH = 0.3

        self.STATE_INFO = {
            'lost': "Can't see full gate, flooring it",
            'found': 'Gate found, aligning',
            'too close': 'Too close to gate, backing up',
        }
        self.state = 'lost'
        self.old_state = ''

    def on_run(self, *args, **kwargs):
        if vision.bottom_prob:
            # If the bottom bar touches the edge of the camera image, we're too
            # close and need to back up a bit. Otherwise, try to make it fill a
            # portion of the camera's width.
            CAM_WIDTH, CAM_HEIGHT = get_camera_size(vision)

            left_x, right_x = vision.bottom_x1, vision.bottom_x2
            if left_x > right_x:
                left_x, right_x = right_x, left_x
            left_prox = left_x / CAM_WIDTH
            right_prox = 1 - (right_x / CAM_WIDTH)

            top_y, bottom_y = vision.bottom_y1, vision.bottom_y2
            if top_y > bottom_y:
                top_y, bottom_y = bottom_y, top_y
            top_prox = top_y / CAM_HEIGHT
            bottom_prox = 1 - (bottom_y / CAM_HEIGHT)

            if left_prox < self.EDGE_PROXIMITY or right_prox < self.EDGE_PROXIMITY or \
                    top_prox < self.EDGE_PROXIMITY or bottom_prox < self.EDGE_PROXIMITY:
                VelocityX(-self.BACKUP_SPEED)()
                self.state = 'too close'

            else:
                self.fast_approach()

        else:
            self.fast_approach()

        if self.state != self.old_state:
            self.logi(self.STATE_INFO[self.state])
            self.old_state = self.state

        if self.pid.finished:
            self.finish()

    def fast_approach(self):
        """ If we're too far from the gate, approach fast. Otherwise, carefully
        align to a fixed distance from the gate.
        """
        if (vision.bottom_prob or \
                (vision.left_prob and vision.right_prob)) and \
                self.width_ratio() >= self.MIN_WIDTH:
            self.pid()
            self.state = 'found'
        else:
            VelocityX(self.APPROACH_SPEED)()
            self.state = 'lost'

    def width_ratio(self):
        CAM_WIDTH, CAM_HEIGHT = get_camera_size(vision)

        if vision.bottom_prob:
            length = abs(vision.bottom_x1 - vision.bottom_x2)
            return length / CAM_WIDTH
        else:
            # We can see both left and right bars
            left_x = (vision.left_x1 + vision.left_x2) / 2
            right_x = (vision.right_x1 + vision.right_x2) / 2
            return (right_x - left_x) / CAM_WIDTH

class AlignSway(Task):
    def on_first_run(self, *args, **kwargs):
        self.pid = PIDLoop(
            input_value=self.height_diff_ratio,
            output_function=VelocityY(),
            target=0,
            deadband=0.01,
            p=20,
            d=10,
        )

    def on_run(self, *args, **kwargs):
        # If we see both bars, try to sway to minimize their height difference
        if vision.left_prob and vision.right_prob and vision.bottom_prob:
            self.pid()

        if self.pid.finished:
            self.finish()

    def height_diff_ratio(self):
        CAM_WIDTH, CAM_HEIGHT = get_camera_size(vision)
        left_height = abs(vision.left_y1 - vision.left_y2)
        right_height = abs(vision.right_y1 - vision.right_y2)
        return (right_height - left_height) / CAM_HEIGHT

class AlignChannel(Task):
    def on_first_run(self, *args, **kwargs):
        self.pids_task = Concurrent(
            AlignHeading(),
            AlignDepth(),
            AlignFore(),
            AlignSway(),
            finite=False,
        )
        self.logi('Starting')

    def on_run(self, *args, **kwargs):
        if not self.pids_task.finished:
            self.pids_task()
        else:
            self.finish()

    def on_finish(self, *args, **kwargs):
        Zero()()
        self.logi('Finished')

class OptimalMission(Task):
    def on_first_run(self, mode=None, main_task=None, *args, **kwargs):
        self.main_task = main_task

    def on_run(self, mode=None, main_task=None, *args, **kwargs):
        if not self.main_task.has_ever_finished:
            self.main_task(mode, *args, **kwargs)
        else:
            self.finish()

    def desiredModules(self):
        return [shm.vision_modules.Navigate]

    def possibleModes(self):
        if self.finished:
            return []
        else:
            return [Mode(name='Full', expectedPoints=800, expectedTime=60)]

align = lambda: Vision(AlignChannel())
barrel_roll = lambda: Vision(StyleBarrelRoll())
unroll = lambda: Vision(StyleBarrelRoll(clockwise=False, period=3, velocity=0))
full = lambda: OptimalMission(main_task=Vision(Sequential(
    # aslam.Target(obj=aslam.world.navigation, finalPosition=np.array([0, -3, 0]), finalTolerance = np.array([0.05, 0.05, 0.05]), observationBoundingBox = (np.array([-2, -2, 0]), np.array([-1, -1, 0]) ) ),
    # aslam.Orient(aslam.world.navigation),
    AlignChannel(),
    StyleSuperSpin(),
)))
basicfull = lambda: Vision(Sequential(AlignChannel(), StyleBasic()))

basic = lambda: Vision(StyleBasic())
pitched = lambda: Vision(StylePitched())
sideways = lambda: Vision(StyleSideways())
upside_down = lambda: Vision(StyleUpsideDown())
loop = lambda: Vision(StyleLoop())
superspin = lambda: StyleSuperSpin()

class Flip180(Task):
    def on_first_run(self, *args, **kwargs):
        #heading = Heading((kalman.heading.get() + 180) % 360, error=1)
        self.flip = Sequential(Pitch(0, error=1), Roll(0, error=1), Timer(1.5), Heading(lambda: kalman.heading.get() + 180, error=1), Timer(1))
    def on_run(self, *args, **kwargs):
        self.flip()
        if self.flip.has_ever_finished:
            self.finish()

est_all = lambda: Sequential(basic(), Heading(90, error=1), pitched(), Heading(270, error=1), sideways(), Heading(90, error=1), upside_down(), Heading(270, error=1), Depth(2.1, error=.1), loop())
