#!/usr/bin/env python3

from vision.modules.base import ModuleBase

class Poster(ModuleBase):
  def process(self, *mats):
    for i, im in enumerate(mats):
      self.post(str(i), im)

if __name__ == '__main__':
    Poster()()
