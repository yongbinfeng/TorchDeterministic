import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from typing import List,Union

import random
import numpy as np
import torch

torch.use_deterministic_algorithms(True)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.input = nn.Linear(5, 1)

    def forward(self, input: Tensor) -> Tensor:
        torch.manual_seed(0)
        inputs = self.input(input)
        perm = torch.randperm(10, dtype=torch.long, device=input.device)
        return perm

model = TestNN()
model = torch.jit.script(model)
torch.jit.save(model, "model.pt")
