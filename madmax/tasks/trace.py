"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


# An individual computation or communication trace
class Trace:
    def __init__(self, name, duration, t_start, in_dep):
        self.trace = {
            "name": name,
            "duration": duration,
            "t_start": t_start,
            "t_end": t_start + duration,
            "in_dep": in_dep,
        }
