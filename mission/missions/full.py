'''
   ___       __                 __     ___  ___ _______
  / _ \___  / /  ___  ___ __ __/ /    |_  |/ _ <  / __/
 / , _/ _ \/ _ \/ _ \(_-</ // / _ \  / __// // / / _ \
/_/|_|\___/_.__/\___/___/\_,_/_.__/ /____/\___/_/\___/

'''

import shm
import aslam

from mission.opt_aux.aux import *
from mission.missions.opt import Opt
from mission.framework.task import Task

# Guessed on the imports, please update for your task.

from mission.missions.aslam_buoys import AllBuoys as Buoys
from mission.missions.navigate import full as Navigation
from mission.missions.torpedoes import Torpedoes
from mission.missions.pipe import OptimizablePipe as Pipe
from mission.missions.bins import OptimizableBins as Bins
from mission.missions.guard import LocaleBounded
#from mission.missions.recovery import mission as Recovery

PipeToBuoys = OptimizableTask(
  name = 'PipeToBuoys',
  cls = lambda: Pipe(grp = shm.buoys_pipe_results),
  startPosition = aslam.world.pipe_to_buoys.position
)

Buoys = OptimizableTask(
  name = 'Buoys',
  cls = Buoys,
  startPosition = aslam.world.red_buoy.position
)

PipeToNavigation = OptimizableTask(
  name = 'PipeToNavigation',
  cls = lambda: Pipe(grp = shm.navigate_pipe_results),
  startPosition = aslam.world.pipe_to_navigation.position
)

Navigation = OptimizableTask(
  name = 'Navigation',
  cls = Navigation,
  startPosition = aslam.world.navigation.position
)

Bins = OptimizableTask(
  name = 'Bins',
  cls = Bins,
  startPosition = aslam.world.bin_one.position
)

Torpedoes = OptimizableTask(
  name = 'Torpedoes',
  cls = Torpedoes,
  startPosition = aslam.world.torpedoes.position
)

#Recovery = OptimizableTask(
#  name = 'Recovery',
#  cls = Recovery,
#  startPosition = aslam.world.recovery_tower.position
#)

restrictions = [
  TopologicalRestriction(beforeTask = 'PipeToBuoys', afterTask = 'Buoys'),
  TopologicalRestriction(beforeTask = 'Buoys', afterTask = 'PipeToNavigation'),
  TopologicalRestriction(beforeTask = 'PipeToNavigation', afterTask = 'Navigation'),
  TopologicalRestriction(beforeTask = 'Navigation', afterTask = 'Bins'),
  TopologicalRestriction(beforeTask = 'Bins', afterTask = 'Torpedoes')
  #TopologicalRestriction(beforeTask = 'Torpedoes', afterTask = 'Recovery')
]

tasks = [
  PipeToBuoys,
  Buoys,
  PipeToNavigation,
  Navigation,
  Bins,
  Torpedoes
  #Recovery
]

Full = LocaleBounded(
  Opt(tasks = tasks, restrictions = restrictions)
)

for task in tasks:
  globals()[task.name] = Opt(tasks = [task], restrictions = [])
