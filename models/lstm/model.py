import torch
import torchvision
import torch.nn.functional as F

from simple_cnn import SimpleCNN

class LSTMPredictor(torch.nn.Module):
    def __init__(self, backbone_cnn, backbone_cnn_output_len):
        super(LSTMPredictor, self).__init__()
        self.backbone_cnn = backbone_cnn
        self.backbone_cnn_output_len = backbone_cnn_output_len
        self.lstm = torch.nn.LSTM(input_size=backbone_cnn_output_len, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(128, num_modes * (2 * predictions_per_mode + 1))

    def forward(self, raster_image, state_vector):
        latent_information = self.backbone_cnn(raster_image, state_vector, latent_output="fc1")
        x, _ = self.lstm(predictions)
        x = self.fc(x)
        return x, probabilities


