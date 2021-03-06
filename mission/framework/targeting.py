from control.pid import DynamicPID
from mission.framework.helpers import call_if_function, within_deadband, PositionalControlManager
from mission.framework.movement import Heading, VelocityY, VelocityX, RelativeToCurrentDepth, RelativeToCurrentHeading
from mission.framework.task import Task


# TODO: Documentation!
class PIDLoop(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pid = None  # type: DynamicPID

    def on_first_run(self, input_value, *args, **kwargs):
        self.pid = DynamicPID()

    def on_run(self, input_value, output_function, target=0, modulo_error=False, deadband=1, p=1, d=0, i=0,
               negate=False, *args, **kwargs):
        # TODO: minimum_output too?
        input_value = call_if_function(input_value)
        target = call_if_function(target)

        output = self.pid.tick(value=input_value, desired=target, p=p, d=d, i=i)
        output_function(-output if negate else output)

        if within_deadband(input_value, target, deadband=deadband, use_mod_error=modulo_error):
            # TODO: Should this zero on finish? Or set to I term?
            self.finish()

class CameraTarget(Task):
    def on_run(self, point, target, deadband=(15, 15), px=None, ix=0, dx=0, py=None, iy=0, dy=0, min_out=None, valid=True):
        if px is None:
            px = self.px_default
        if py is None:
            py = self.py_default

        if valid:
            point = call_if_function(point)
            target = call_if_function(target)
            self.pid_loop_x(input_value=point[0], p=px, i=ix, d=dx, target=target[0], deadband=deadband[0], min_out=min_out)
            self.pid_loop_y(input_value=point[1], p=py, i=iy, d=dy, target=target[1], deadband=deadband[1], min_out=min_out)
        else:
           self.stop()

        if self.pid_loop_x.finished and self.pid_loop_y.finished:
            # TODO: Should the output be zeroed on finish?
            self.finish()

class ForwardTarget(CameraTarget):
    def on_first_run(self, *args, **kwargs):
        self.pid_loop_x = PIDLoop(output_function=VelocityY(), negate=True)
        self.pid_loop_y = PIDLoop(output_function=RelativeToCurrentDepth(), negate=True)
        self.px_default = 0.001
        self.py_default = 0.001
        PositionalControlManager().set(0)

    def stop(self):
        VelocityY(0)()
        RelativeToCurrentDepth(0)()

class DownwardTarget(CameraTarget):
    def on_first_run(self, *args, **kwargs):
        # x-axis on the camera corresponds to sway axis for the sub
        self.pid_loop_x = PIDLoop(output_function=VelocityY(), negate=True)
        self.pid_loop_y = PIDLoop(output_function=VelocityX(), negate=False)
        self.px_default = 0.0005
        self.py_default = 0.001
        PositionalControlManager().set(0)

    def stop(self):
        VelocityY(0)()
        VelocityX(0)()

class HeadingTarget(CameraTarget):
    def on_first_run(self, *args, **kwargs):
        self.pid_loop_y = PIDLoop(output_function=RelativeToCurrentDepth(), negate=True)
        self.pid_loop_x = PIDLoop(output_function=RelativeToCurrentHeading(), negate=True)
        self.px_default = 0.01
        self.py_default = 0.001

    def stop(self):
        RelativeToCurrentDepth(0)()
        RelativeToCurrentHeading(0)()

class DownwardAlign(Task):
    def on_first_run(self, *args, **kwargs):
        self.pid_loop_heading = PIDLoop(output_function=RelativeToCurrentHeading(), modulo_error=True)

    def on_run(self, angle, deadband=3, p=.001, i=0, d=0, target=0, modulo_error=True):
        angle = call_if_function(angle)
        self.pid_loop_heading(input_value=angle, p=p, i=i, d=d, target=target, deadband=deadband, negate=True)

        if self.pid_loop_heading.finished:
            self.finish()
