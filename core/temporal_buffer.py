# core/temporal_buffer.py

import numpy as np
from collections import deque


class RollingBuffer:
    def __init__(self, length):
        self.length = length
        self.buffer = deque(maxlen=length)

    def add(self, z):
        self.buffer.append(z)

    def ready(self):
        return len(self.buffer) == self.length

    def get(self):
        return np.array(self.buffer)