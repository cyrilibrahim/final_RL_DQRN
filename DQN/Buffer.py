import random
import numpy as np
from StateTransition import Transition


class Buffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    # Similar to a queue with fixed size
    # FIFO first in first out
    def push(self, state, action, next_state, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)

        self.buffer.append(Transition(state, action, reward, next_state, done))

    # Number of not None
    def hasAtLeast(self, size):
        return len(self.buffer) >= size

    # Sample nb_samples entries
    def sample(self, nb_samples):
        return random.sample(self.buffer, nb_samples)
