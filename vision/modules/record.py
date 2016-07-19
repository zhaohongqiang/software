#!/usr/bin/env python3

from vision.modules.logger import VideoWriter
from vision.modules.base import ModuleBase

class Record(ModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = [VideoWriter(direction) for direction in self.directions]

    def process(self, *mats):
        for i, im in enumerate(mats):
            self.writers[i].log_image(im)

if __name__ == '__main__':
    Record()()
