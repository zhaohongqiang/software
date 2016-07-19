'''

Optimal Mission Runner

Introduction:

Usage:

Tasks must expose the following interface:

  possibleModes :: () -> [mode]
  
  --> This means tasks need to *keep* their own state. Finished tasks should return an empty list (no possible modes remaining).
  
Tasks are passed the following parameters on run:

  mode :: string (the name of the mode)

Caveats / To Know:

Task instances may be re-run after periods of inactivity (not being run).
This can cause issues with quasi-time-dependent state, e.g. PID loops calculating dts.
Suggested fix: check self.last_run in on_run and reset such state if too much time has passed.

'''

from functools import *

from mission.opt_aux.aux import *
from mission.framework.task import *
from mission.framework.combinators import *
from mission.framework.movement import *
from mission.framework.primitive import *

from termcolor import colored

import shm, time, opt, numpy

all_modules = [
  shm.vision_modules.Bins,
  shm.vision_modules.Buoys,
  shm.vision_modules.Debug,
  shm.vision_modules.Egomotion,
  shm.vision_modules.Navigate,
  shm.vision_modules.Pipes,
  shm.vision_modules.Recovery,
  shm.vision_modules.Sonar,
  shm.vision_modules.Stereo,
  shm.vision_modules.Torpedoes
]

def assertModules(modules, log):
  for v in all_modules:
    prev = v.get()
    if v in modules:
      v.set(True)
      if not prev:
        log('Now enabling vision module: {}'.format(colored(v.__name__, 'green', attrs = ['bold'])))
    else:
      v.set(False)
      if prev:
        log('Now disabling vision module: {}'.format(colored(v.__name__, 'red', attrs = ['bold'])))

def execute(instance, taskPlan):
  instance(
    mode = taskPlan.mode.name
    )

def prettify(taskPlan):
  return ' => '.join(colored(task.taskName, 'green', attrs = ['bold']) + ' @ ' + colored(task.mode.name, 'cyan', attrs = ['bold']) for task in taskPlan)

def equivalent(planA, planB):
  return len(planA) == len(planB) and all([x.taskName == y.taskName and x.mode.name == y.mode.name for (x, y) in zip(planA, planB)])

class Opt(Task):
  def __init__(self, tasks, restrictions, maxExceptionCount = 5):
    super().__init__()
    self.tasks = tasks
    self.restrictions = restrictions
    self.maxExceptionCount = maxExceptionCount
    self.namesToIDs = {task.name: ind for (ind, task) in enumerate(self.tasks)}
    self.idsToNames = {ind: task.name for (ind, task) in enumerate(self.tasks)}
    self.namesToExceptionCounts = {task.name: 0 for task in self.tasks}
    self.taskExecutable = {task.name: True for task in self.tasks}
    self.instances = {task.name: task.cls() for task in self.tasks}

  def on_first_run(self):
    self.initialization = time.time()
    self.lastTask = None
    self.lastMode = None
    self.lastPlan = None

  def on_run(self):
    start = time.time()

    # Construct full distance map
    positions = {self.namesToIDs[task.name] : task.startPosition() for task in self.tasks}
    fullDistanceMap = opt.FullDistanceMap()
    for taskInd, task in enumerate(self.tasks):
      distanceMap = opt.DistanceMap()
      for otherInd, other in enumerate(self.tasks):
        distanceMap[otherInd] = numpy.linalg.norm(positions[otherInd] - positions[taskInd])
      fullDistanceMap[taskInd] = distanceMap

    # Construct optimizable tasks
    possibleTasks = set()
    optimizableTaskList = opt.OptimizableTaskList()
    modes = {self.namesToIDs[task.name] : self.instances[task.name].possibleModes() for task in self.tasks}
    for taskInd, task in enumerate(self.tasks):
      if not self.taskExecutable[task.name]:
        continue
      modeList = opt.ModeList()
      for modeInd, mode in enumerate(modes[taskInd]):
        modeList.append(opt.Mode(modeInd, mode.expectedPoints, mode.expectedTime))
      if len(modes[taskInd]) > 0:
        possibleTasks.add(self.idsToNames[taskInd])
        optimizableTaskList.append(opt.OptimizableTask(taskInd, modeList))

    # Construct topological restrictions
    topologicalRestrictionList = opt.TopologicalRestrictionList()
    for restriction in self.restrictions:
      if restriction.beforeTask in possibleTasks: # TODO FIXME better way, this doesn't really check completion status?
        topologicalRestrictionList.append(opt.TopologicalRestriction(self.namesToIDs[restriction.beforeTask], self.namesToIDs[restriction.afterTask]))

    remainingTime = (60. * 15) - (time.time() - self.initialization)
    self.logv('Remaining time: {}s'.format(remainingTime))

    res = opt.optimize(fullDistanceMap, remainingTime, optimizableTaskList, topologicalRestrictionList)

    end = time.time()
    self.logv('Generated, validated, and scored possible execution plans in {} seconds.'.format(end - start))

    if len(res) > 0:
      plan = []
      for val in res:
        plan.append(TaskPlan(self.idsToNames[val.id()], modes[val.id()][val.mode().id()]))
      if self.lastPlan is None or not equivalent(plan, self.lastPlan):
        self.logi('New optimal execution plan: {}.'.format(prettify(plan)))
      self.lastPlan = plan
      taskPlan = plan[0]
      if taskPlan.taskName != self.lastTask or taskPlan.mode.name != self.lastMode:
          self.instances[taskPlan.taskName] = self.tasks[self.namesToIDs[taskPlan.taskName]].cls()
          self.logi('Reinstantiated task: {} due to reason: switched.'.format(taskPlan.taskName))
          self.logi('Switched to: task {} with mode {}.'.format(colored(taskPlan.taskName, 'green', attrs = ['bold']), colored(taskPlan.mode.name, 'cyan', attrs = ['bold'])))
      self.lastTask = taskPlan.taskName
      self.lastMode = taskPlan.mode.name

      assertModules(self.instances[taskPlan.taskName].desiredModules(), self.logi)
      try: 
        execute(self.instances[taskPlan.taskName], taskPlan)
      except Exception as e:
        self.namesToExceptionCounts[taskPlan.taskName] += 1
        if self.namesToExceptionCounts[taskPlan.taskName] < self.maxExceptionCount:
          self.logw('Task {} threw exception: {}! Exception {} of {} before that task is killed!'.format(taskPlan.taskName, \
            e, self.namesToExceptionCounts[taskPlan.taskName], self.maxExceptionCount))
        else:
          self.loge('Task {} threw exception: {}! Task has reached exception threshold, will no longer be attempted!'.format( \
            taskPlan.taskName, e))
          self.taskExecutable[taskPlan.taskName] = False

    else:
      self.finish()

  def on_finish(self):
    self.logv('Optimal mission complete!')
