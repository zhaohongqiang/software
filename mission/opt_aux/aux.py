'''

Optimal Mission Runner Types / Auxiliary

Separated out for ease-of-import.

'''

from collections import namedtuple
from enum import Enum

TaskPlan = namedtuple('TaskPlan', [
  'taskName',
  'mode'
  ])

ExecutionPlanStat = namedtuple('ExecutionPlanStat', [
  'expectedPoints',
  'expectedTime'
  ])

'''
Mode: A way in which a task can be run.

Examples:
  -> Various passage options for navigation.
  -> Firing one vs. two torpedoes for torpedoes.
  -> Dropping on one / two / specific bins for bins.

Modes must specify:
  name :: string (passed to the task)
  expectedPoints :: double
  expectedTime :: double (seconds)

expectedPoints should be equal to the point total given by the rules multiplied by the expected chance of success of the task.
'''

Mode = namedtuple('Mode', [
  'name',
  'expectedPoints',
  'expectedTime'
  ])

'''
OptimizableTask: A task that can be run by the optimal mission runner.

OptimizableTasks must specify:
  name :: string
  cls :: class (must be subtask of Task)
  startPosition :: () -> numpy 3-vector (north, east, depth)
'''

OptimizableTask = namedtuple('OptimizableTask', [
  'name',
  'cls',
  'startPosition'
  ])

'''
Topological Restriction: A required ordering of tasks.

TopologicalRestrictions must specify
  beforeTask :: string
  afterTask  :: string
'''

TopologicalRestriction = namedtuple('TopologicalRestriction', [
  'beforeTask',
  'afterTask'
  ])
