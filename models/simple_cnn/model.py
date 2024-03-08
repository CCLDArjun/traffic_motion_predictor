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
        probabilities, predictions = torch.split(x, [self.num_modes, x[0].shape[0] - self.num_modes], dim=1)

        predictions = torch.reshape(predictions, (-1, self.num_modes, self.predictions_per_mode, 2))
        probabilities = F.softmax(probabilities, dim=1)

        return predictions, probabilities

def _trajectory_distance(pred, target):
    return torch.norm(pred - target)

def _torch_empty_factory(tensor):
    def F(*args, **kwargs):
        return tensor.new_empty(*args, **kwargs)
    return F

def _torch_zeros_factory(tensor):
    def F(*args, **kwargs):
        return tensor.new_zeros(*args, **kwargs)
    return F

def loss_function(predictions, probabilities, target_prediction, prediction_loss_weight=torch.tensor(1.0)):
    torch_empty = _torch_empty_factory(predictions)
    torch_zeros = _torch_zeros_factory(predictions)

    modes = predictions[0].shape[0]
    batch_losses = torch_empty(predictions.shape[0], 1)

    for batch_i, batch in enumerate(predictions):
        distances = torch_empty(modes, 1)
        for mode_i, mode in enumerate(batch):
            distances[mode_i] = _trajectory_distance(
                predictions[batch_i][mode_i],
                target_prediction[batch_i]
            )

        _, closest_trajectory_i = torch.min(distances, dim=0)
        closest_trajectory = predictions[batch_i][closest_trajectory_i].squeeze(0)

        l1_loss = F.smooth_l1_loss(closest_trajectory, target_prediction[batch_i])

        target_probability = torch_zeros(modes)
        target_probability[closest_trajectory_i] = 1

        confidence_loss = F.cross_entropy(probabilities[batch_i], target_probability)


        print(f"l1_loss: {l1_loss}, confidence_loss: {confidence_loss}")
        loss = (prediction_loss_weight * l1_loss) + confidence_loss

        batch_losses[batch_i] = loss

    return torch.mean(batch_losses)

