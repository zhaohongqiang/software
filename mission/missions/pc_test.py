from framework.position import MoveX, MoveY
from framework.combinators import Sequential

task = Sequential(MoveX(1), MoveY(1))
movex = MoveX(1)
movey = MoveY(1)
#task = Sequential(MoveY(0.10), MoveX(0.60), MoveY(-0.20), MoveX(-0.12), MoveY(0.10), MoveX(-0.1))
