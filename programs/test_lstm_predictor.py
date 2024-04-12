import sys
sys.path.append("../")

import torch

from models.simple_cnn import SimpleCNN, loss_function
from models.lstm import LSTMPredictor, test

backbone_cnn = SimpleCNN(25, 12, state_vector_size=12)
model = LSTMPredictor(backbone_cnn, 4096, 25)
batches = 2

raster_image = torch.rand(batches, 3, 500, 500)
state_vector = torch.rand(batches, 12)

predictions, probabilites = model(raster_image, state_vector)
print(f"{predictions.shape=}")
print(f"{probabilites.shape=}")

test()

