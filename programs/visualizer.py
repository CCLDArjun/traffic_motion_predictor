import sys
sys.path.append("../")

import torch
import matplotlib.pyplot as plt

from dataset import ChunkReader
from dataset.simple_cnn_dataset import denormalize_xy, normalize
import random

d = ChunkReader("./chunks/")

# choose a random data point
x = 4000
raster_image, _, gt_xy  = d[x]

# plot the raster image
raster_image = raster_image.permute(0, 2, 3, 1).squeeze() / 255.
plt.imshow(raster_image)

# plot the target trajectory

gt_xy *= 500
plt.plot(gt_xy[:, 0], gt_xy[:, 1], 'ro')

print(gt_xy)
plt.savefig("raster_image.png")

