import torch
import torchvision
import torch.nn.functional as F

from simple_cnn import SimpleCNN

class LSTMPredictor(torch.nn.Module):
    def __init__(self, traffic_encoder, traffic_encoder_output_len, hidden_size=128):
        super(LSTMPredictor, self).__init__()
        self.traffic_encoder = traffic_encoder
        self.traffic_encoder_output_len = traffic_encoder_output_len
        self.lstm = torch.nn.LSTM(
            input_size=traffic_encoder_output_len,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc_out = torch.nn.Linear(hidden_size, 2 * num_modes)
        self.fc_hidden_in = torch.nn.Linear(traffic_encoder_output_len, hidden_size)
        self.fc_input_in = torch.nn.Linear(traffic_encoder_output_len, hidden_size)

    def forward(self, raster_image, state_vector, points_to_predict=12):
        predictions = torch.empty((points_to_predict, 2 * num_modes))

        encoded = self.traffic_encoder(raster_image, state_vector, latent_output="fc1")
        hidden = self.fc_in(encoded)
        cell = torch.zeros_like(hidden)

        for i in range(points_to_predict):
            output, (hidden, cell) = self.lstm(, (hidden, cell))
            output = self.fc_out(output)

            predictions[i] = output

        return predictions

