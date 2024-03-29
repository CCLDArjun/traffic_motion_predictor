import sys
sys.path.append("../")

import torch
import matplotlib.pyplot as plt

from dataset import ChunkReader
import random

d = ChunkReader("./chunks/")

# choose a random data point
x = random.randrange(0, len(d))
sample = d[x]

# plot the raster image


