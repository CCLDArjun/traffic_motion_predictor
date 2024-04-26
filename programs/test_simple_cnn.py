import sys
sys.path.append("..")

import torch
from models.simple_cnn import SimpleCNN
import dataset

#d = dataset.NuScenesDataset("../data/sets/v1.0-mini")
#input_image = torch.from_numpy(d[0]['top_down_repr']).unsqueeze(0)
#input_image = input_image.permute(0, 3, 1, 2).float()

model = SimpleCNN(5, 12)
pred = model(torch.randn((1, 3, 500, 500)), torch.rand(1, 4))
print("predictions:", pred[0].shape, "confidence:", pred[1].shape)

