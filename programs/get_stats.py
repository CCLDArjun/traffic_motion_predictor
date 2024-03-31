import sys
sys.path.append("../")

import torch

from dataset import ChunkReader
from dataset.simple_cnn_dataset import denormalize_xy, normalize
import random

d = ChunkReader("./chunks/")
dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True)

class MetricAccum():
    def __init__(self, to_save):
        self.tensors = {}
        self.to_save = to_save

    def update(self, d):
        for k, v in d.items():
            if k not in self.to_save: continue

            v = v.flatten()
            if k not in self.tensors:
                self.tensors[k] = v
            else:
                self.tensors[k] = torch.cat((self.tensors[k], v))

accum = MetricAccum(["velocity", "acceleration", "heading_change_rate", "xy_in", "gt_xy"])

for i in range(len(d)):
    data = d[i]
    raster_image, state_vector, gt_xy = data

    velocity = state_vector[0]
    acceleration = state_vector[1]
    heading_change_rate = state_vector[2]
    xy_in = state_vector[3:]

    metric_dict = {
        "raster_image": raster_image,
        "state_vector": state_vector,
        "gt_xy": gt_xy,
        "velocity": velocity,
        "acceleration": acceleration,
        "heading_change_rate": heading_change_rate,
        "xy_in": xy_in,
        "gt_xy": gt_xy,
    }

    accum.update(metric_dict)

for k, v in accum.tensors.items():
    print(f"Stats of {k}:")
    print(f"     mean: {v.mean()}")
    print(f"      min: {v.min()}")
    print(f"      max: {v.max()}")
    print(f"      std: {v.std()}")
    print(f" variance: {v.var()}")

