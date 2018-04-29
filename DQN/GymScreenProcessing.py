import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np

class GymScreenProcessing(object):

    def __init__(self, env, use_cuda):
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.Tensor = FloatTensor
        self.env = env

        self.resize = T.Compose([T.ToPILImage(),
                            T.Resize((84,84), interpolation=Image.CUBIC),
                            T.Grayscale(),
                            T.ToTensor()])

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose(
            (2, 0, 1))  # transpose into torch order (CHW)

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0).type(self.Tensor)
