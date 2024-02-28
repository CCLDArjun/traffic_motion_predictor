"""
Inspired by MTP: https://arxiv.org/pdf/1809.10732.pdf
"""

import torch
import torchvision
import torch.nn.functional as F

def construct_base_cnn():
    base_cnn = torchvision.models.mobilenet_v2(pretrained=True)
    base_cnn_raster_size = base_cnn.classifier[1].in_features
    base_cnn.classifier = torch.nn.Identity() # remove the last layer
    return base_cnn, base_cnn_raster_size

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_modes, predictions_per_mode, state_vector_size=4, state_vector_replication=10, fc1_size=4096):
        super(SimpleCNN, self).__init__()
        self.base_cnn, base_cnn_raster_size = construct_base_cnn()
        self.fc1 = torch.nn.Linear(base_cnn_raster_size + (state_vector_size * state_vector_replication), fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, num_modes * (2 * predictions_per_mode + 1)) # 2 for x, y and 1 for confidence

        self.num_modes = num_modes
        self.state_vector_replication = state_vector_replication
        self.predictions_per_mode = predictions_per_mode

    def forward(self, raster_image, state_vector, latent_output=""):
        raster_features = self.base_cnn(raster_image)

        state_vector = state_vector.repeat(1, self.state_vector_replication)
        x = torch.cat([raster_features, state_vector], dim=1)
        x = F.relu(self.fc1(x))

        if latent_output == "fc1":
            return fc1

        x = self.fc2(x)

        # each mode has a probability so the length is num_modes
        probabilities, predictions = torch.split(x, [self.num_modes, len(x[0]) - self.num_modes], dim=1)

        predictions = torch.reshape(predictions, (-1, self.num_modes, 2 * self.predictions_per_mode))
        probabilities = F.softmax(probabilities, dim=1)

        return predictions, probabilities

@torch.jit.script
def _trajectory_distance(pred, target):
    return torch.norm(pred - target)

@torch.jit.script
def loss_function(predictions, probabilities, target_prediction, prediction_loss_weight=torch.tensor(1.0)):
    modes = len(predictions[0])
    batch_losses = torch.empty(len(predictions), 1)

    for i, batch in enumerate(predictions):
        distances = torch.empty(modes, 1)
        for i, mode in enumerate(batch):
            distances[i] = _trajectory_distance(predictions[batch][mode], target_prediction[batch])
        closest_trajectory, index = torch.min(distances, dim=0)

        l1_loss = F.smooth_l1_loss(closest_trajectory, target_prediction[i])

        target_probability = torch.zeros(modes)
        target_probability[index] = 1

        confidence_loss = F.cross_entropy(probabilities[i], target_probability)

        loss = (prediction_loss_weight * l1_loss) + confidence_loss

        batch_losses[i] = loss

    return torch.mean(batch_losses)

