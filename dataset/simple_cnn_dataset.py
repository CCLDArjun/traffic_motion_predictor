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


class SimpleCNNDataset(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        cache_size = kwargs.pop("cache_size", 10_000_000)

        super().__init__(*args, **kwargs)

        self.__getitem__ = functools.lru_cache(maxsize=cache_size)(self.__getitem__)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        state_vector = torch.tensor([
            normalize(data["velocity"], MIN_VELOCITY, MAX_VELOCITY),
            normalize(data["acceleration"], MIN_ACCELERATION, MAX_ACCELERATION),
            normalize(data["heading_change_rate"], MIN_HEADING_CHANGE_RATE, MAX_HEADING_CHANGE_RATE),
        ])

        state_vector[state_vector.isnan()] = 0

        xy_in = torch.flatten(torch.Tensor(data["past"]["agent_xy_global"]))
        xy_in = normalize(xy_in, MIN_XY, MAX_XY)

        state_vector = torch.cat([state_vector, xy_in])
        agent_rast = torch.Tensor(data["agent_rast"])

        return (
            agent_rast.unsqueeze(0).permute(0, 3, 1, 2).float(),
            state_vector.float(),
            normalize(torch.from_numpy(data["future"]["agent_xy_global"]), MIN_XY, MAX_XY),
        )


def normalization_values():
    maxes = [0, 0, 0]
    mins = [0, 0, 0]
    max_xy = 0
    min_xy = 0
    i = 0
    for data in d:
        if i % 50 == 0:
            print(i)
        for i in range(3):
            maxes[i] = max(maxes[i], data[1][i])
            mins[i] = min(mins[i], data[1][i])

        max_xy = max(max_xy, data[1][3:].max())
        min_xy = min(max_xy, data[1][3:].min())

        max_xy = max(max_xy, data[2].max())
        min_xy = min(max_xy, data[2].min())

    print("MAXES: ", maxes)
    print("MINS: ", mins)
    print("XY: ", min_xy, max_xy)
