# core/state_manager.py

import numpy as np
from collections import defaultdict


class StreamState:
    def __init__(self, groups, cold_start):
        self.groups = groups
        self.cold_start = cold_start

        self.buffer = defaultdict(list)
        self.initialized = False

        self.healthy = {}
        self.excluded = {}

    def update_buffer(self, values_dict):
        for g in ["P", "T", "F"]:
            for s in self.groups[g]:
                self.buffer[s].append(values_dict[s])

    def check_cold_start_ready(self):
        return all(len(self.buffer[s]) >= self.cold_start
                   for g in ["P", "T", "F"]
                   for s in self.groups[g])

    def finalize_static_filter(self, eps=1e-6):
        for g in ["P", "T", "F"]:
            healthy = []
            excluded = []

            for s in self.groups[g]:
                std_val = np.std(self.buffer[s][:self.cold_start])
                if std_val < eps:
                    excluded.append(s)
                else:
                    healthy.append(s)

            self.healthy[g] = healthy
            self.excluded[g] = excluded

        self.initialized = True