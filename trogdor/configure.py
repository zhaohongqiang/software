#!/usr/bin/python

from build import ninja_common

build = ninja_common.Build('trogdor')

build.install('trogdor', f='trogdor/trogdor.sh')
