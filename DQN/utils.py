import numpy as np
import torch
from torch.autograd import Variable


def img_to_tensor(obs, tensor_type=torch.cuda.FloatTensor):
    obst = np.transpose(obs, (0, 3, 1, 2))
    img = torch.from_numpy(obst)
    return img.type(tensor_type)