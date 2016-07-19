from mission.framework.task import *
from mission.framework.position import *
from mission.framework.movement import *
from mission.framework.combinators import *

class ConfigurableWiggle(Task):
    def on_first_run(self, wiggle_amount, num_wiggles, wiggle_x, wiggle_y, wiggle_z):
        tasks = []
        if wiggle_x:
          tasks.append(MoveXRough(-wiggle_amount / 2.))
        if wiggle_y:
          tasks.append(MoveYRough(-wiggle_amount / 2.))
        if wiggle_z:
          tasks.append(RelativeToInitialDepth(- wiggle_amount / 2.))
        for num in range(num_wiggles):
          negated = num % 2 == 1
          amount  = (-1 if negated else 1) * wiggle_amount
          if wiggle_x: tasks.append(MoveXRough(amount))
          if wiggle_y: tasks.append(MoveYRough(amount))
          if wiggle_z: tasks.append(RelativeToInitialDepth(amount))
        if wiggle_x:
          tasks.append(MoveXRough(wiggle_amount / 2))
        if wiggle_y:
          tasks.append(MoveYRough(wiggle_amount / 2))
        if wiggle_z:
          tasks.append(RelativeToInitialDepth(wiggle_amount / 2))

        self.task = Sequential(*tasks)

    def on_run(self, *args, **kwargs):
        if not self.task.finished:
            self.task()
        else:
            self.finish()

BigWiggle     = lambda x, y, z: ConfigurableWiggle(0.2, 10, x, y, z)
MediumWiggle  = lambda x, y, z: ConfigurableWiggle(0.1, 5, x, y, z)
SmallWiggle   = lambda x, y, z: ConfigurableWiggle(0.05, 3, x, y, z)

SmallXWiggle    = SmallWiggle(True, False, False)
SmallYWiggle    = SmallWiggle(False, True, False)
SmallZWiggle    = SmallWiggle(False, False, True)
SmallXYWiggle   = SmallWiggle(True, True, False)
SmallXZWiggle   = SmallWiggle(True, False, True)
SmallYZWiggle   = SmallWiggle(False, True, True)
SmallXYZWiggle  = SmallWiggle(True, True, True)

MediumXWiggle    = MediumWiggle(True, False, False)
MediumYWiggle    = MediumWiggle(False, True, False)
MediumZWiggle    = MediumWiggle(False, False, True)
MediumXYWiggle   = MediumWiggle(True, True, False)
MediumXZWiggle   = MediumWiggle(True, False, True)
MediumYZWiggle   = MediumWiggle(False, True, True)
MediumXYZWiggle  = MediumWiggle(True, True, True)

BigXWiggle    = BigWiggle(True, False, False)
BigYWiggle    = BigWiggle(False, True, False)
BigZWiggle    = BigWiggle(False, False, True)
BigXYWiggle   = BigWiggle(True, True, False)
BigXZWiggle   = BigWiggle(True, False, True)
BigYZWiggle   = BigWiggle(False, True, True)
BigXYZWiggle  = BigWiggle(True, True, True)
