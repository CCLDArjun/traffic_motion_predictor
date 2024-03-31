import functools

import torch

from .nuscenes import NuScenesDataset

MIN_XY = 842
MAX_XY = 2733

MAX_VELOCITY = 20
MIN_VELOCITY = 0

MAX_ACCELERATION = 10
MIN_ACCELERATION = -10

MAX_HEADING_CHANGE_RATE = 0.42
MIN_HEADING_CHANGE_RATE = -0.26


def normalize(value, min, max):
    return (value - min) / (max - min)

def denormalize(value, min, max):
    return value * (max - min) + min

denormalize_xy = functools.partial(denormalize, min=MIN_XY, max=MAX_XY)

class SimpleCNNDataset(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        cache_size = kwargs.pop("cache_size", 10_000_000)
        self.normalize = kwargs.pop("normalize", True)
        if self.normalize:
            self.normalize = self.normalize
        else:
            self.normalize = self.empty_normalize

        super().__init__(*args, **kwargs)

    def empty_normalize(self, value, min, max):
        return value
    def normalize(self, value, min, max):
        return normalize(value, min, max)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        state_vector = torch.tensor([
            self.normalize(data["velocity"], MIN_VELOCITY, MAX_VELOCITY),
            self.normalize(data["acceleration"], MIN_ACCELERATION, MAX_ACCELERATION),
            self.normalize(data["heading_change_rate"], MIN_HEADING_CHANGE_RATE, MAX_HEADING_CHANGE_RATE),
        ])

        state_vector[state_vector.isnan()] = 0

        xy_in = torch.flatten(torch.Tensor(data["past"]["agent_xy_global"]))
        xy_in = self.normalize(xy_in, MIN_XY, MAX_XY)

        state_vector = torch.cat([state_vector, xy_in])
        agent_rast = torch.Tensor(data["agent_rast"])

        return (
            agent_rast.unsqueeze(0).permute(0, 3, 1, 2).float(),
            state_vector.float(),
            self.normalize(torch.from_numpy(data["future"]["agent_xy_global"]), MIN_XY, MAX_XY),
        )


