from collections import deque


class StackFrame(object):

    def __init__(self, max_frames):
        self.frames = deque([], maxlen=max_frames)
        self.max_frames = max_frames

    def add_frame(self, frame):
        self.frames.append(frame)

    def reset_stack(self):
        self.frames = deque([], maxlen=self.max_frames)


    def is_ready(self):
        return len(self.frames) >= self.max_frames


    def get_frames(self):
        assert self.is_ready()
        return list(self.frames)

