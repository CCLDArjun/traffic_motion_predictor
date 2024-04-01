import functools

import torch

from .nuscenes import NuScenesDataset

# mini dataset
#MIN_XY = 842
#MAX_XY = 2733
#
#MAX_VELOCITY = 20
#MIN_VELOCITY = 0
#
#MAX_ACCELERATION = 10
#MIN_ACCELERATION = -10
#
#MAX_HEADING_CHANGE_RATE = 0.42
#MIN_HEADING_CHANGE_RATE = -0.26

# full dataset

MIN_XY = 117
MAX_XY = 3332

MAX_VELOCITY = 24
MIN_VELOCITY = 0

MAX_ACCELERATION = 25
MIN_ACCELERATION = 0

MAX_HEADING_CHANGE_RATE = 1.5
MIN_HEADING_CHANGE_RATE = 0

"""
Stats of gt_xy:
     mean: 1208.4489311765315
      min: 117.829
      max: 3327.133
      std: 524.2514530969484
 variance: 274839.58607426187
Stats of velocity:
     mean: 5.806914329528809
      min: 0.0
      max: 23.447195053100586
      std: 3.8555655479431152
 variance: 14.865385055541992
Stats of acceleration:
     mean: 0.08666366338729858
      min: -24.484315872192383
      max: 21.7139892578125
      std: 1.4933940172195435
 variance: 2.2302255630493164
Stats of heading_change_rate:
     mean: 0.0002595177502371371
      min: -0.9776309728622437
      max: 1.223345160484314
      std: 0.095574751496315
 variance: 0.009134532883763313
Stats of xy_in:
     mean: 1207.942138671875
      min: 117.88099670410156
      max: 3331.178955078125
      std: 523.8219604492188
 variance: 274389.46875
 """

def normalize(value, min, max):
    return (value - min) / (max - min)

def denormalize(value, min, max):
    return value * (max - min) + min

denormalize_xy = functools.partial(denormalize, min=MIN_XY, max=MAX_XY)

class SimpleCNNDataset(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        cache_size = kwargs.pop("cache_size", 10_000_000)

        normalize = kwargs.pop("normalize", True)
        if normalize:
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
        agent_rast = torch.Tensor(data["input_image"])

        return (
            agent_rast.unsqueeze(0).permute(0, 3, 1, 2).float(),
            state_vector.float(),
            self.normalize(torch.from_numpy(data["future"]["agent_xy_global"]), MIN_XY, MAX_XY),
        )


