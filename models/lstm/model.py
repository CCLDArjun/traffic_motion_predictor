"""
This file contains the model definition for the LSTM predictor.
Inspired by: https://arxiv.org/pdf/1808.05819.pdf
"""

import torch
import torchvision
import torch.nn.functional as F
from ..simple_cnn import SimpleCNN

class LSTMPredictor(torch.nn.Module):
    def __init__(self, traffic_encoder, traffic_encoder_output_len, num_modes, hidden_size=128):
        super(LSTMPredictor, self).__init__()
        self.traffic_encoder = traffic_encoder
        self.traffic_encoder_output_len = traffic_encoder_output_len
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=10,
            batch_first=True,
        )

        self.num_modes = num_modes
        self.fc_in = torch.nn.Linear(traffic_encoder_output_len, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, 3 * num_modes)

    def forward(self, raster_image, state_vector, points_to_predict=12):
        device = raster_image.device
        batches = raster_image.shape[0]

        backbone_encoded = self.traffic_encoder(
            raster_image,
            state_vector,
            latent_output="fc1"
        )

        encoded_traffic = F.relu(self.fc_in(backbone_encoded))

        # (num_layers, batch, hidden_size), but (num_layers, hidden_size) for unbatched
        hidden = torch.zeros(
            (self.lstm.num_layers, encoded_traffic.shape[0], self.lstm.hidden_size)
        ).squeeze().to(device)
        cell = torch.zeros_like(hidden).to(device)

        encoded_traffic = encoded_traffic.unsqueeze(1).repeat(1, points_to_predict, 1)

        #print(f"{encoded_traffic.shape=} {hidden.shape=} {cell.shape=}")
        lstm_output, (hidden, cell) = self.lstm(encoded_traffic, (hidden, cell))
        fc_output = self.fc_out(lstm_output)

        predictions = self.create_predictions(fc_output)
        probabilities = self.create_probabilities(fc_output)

        return predictions, probabilities
    
    def create_predictions(self, fc_output):
        predictions = fc_output[:, :, :50]

        # we want predictions to be (B, 25, 12, 2)
        # cant just reshape(B, 25, 12, 2) otherwise it messes up the order of the 
        # predictions as they are from the lstm passes
        predictions = predictions.transpose(1, 2) # (B, 12, 50) -> (B, 50, 12)
        predictions = predictions.chunk(25, dim=1) # (B, 50, 12) -> 25 * (B, 2, 12)
        predictions = torch.stack(predictions, dim=1) # 25 * (B, 12, 2) -> (B, 25, 2, 12)
        predictions = predictions.transpose(3, 2) # (B, 25, 2, 12) -> (B, 25, 12, 2)

        return predictions

    def create_probabilities(self, fc_output):
        probabilities = fc_output[:, :, 50:]

        # TODO: check if correct, made by chatgpt
        # only care about probabilities from the last prediction
        probabilities = probabilities[:, -1, :]
        probabilities = F.softmax(probabilities, dim=1)
        return probabilities

def test():
    torch.set_printoptions(precision=2)
    torch.set_printoptions(sci_mode=False)
    backbone_cnn = SimpleCNN(25, 12, state_vector_size=12)
    model = LSTMPredictor(backbone_cnn, 4096, 25)
    batches = 2
    points_per_traj = 3
    modes = 25

    raster_image = torch.rand(batches, 3, 500, 500)
    state_vector = torch.rand(batches, 12)

    fc_out = torch.randperm(batches * points_per_traj * modes * 3).reshape(batches, points_per_traj, modes * 3)

    predictions = model.create_predictions(fc_out)

    """
    Illustration of what we're checking:
                self.fc_out(lstm_output[0])
                v 
    fc_out = [[[a, b, c, d, m, n], [e, f, g, h, o, p], [i, j, k, l, q, r]]]
                                    ^ self.fc_out(lstm_output[1])
    Want to make sure that predictions are:
    [[[a, b],
      [e, f],
      [i, j]],

     [[c, d],
      [g, h],
      [k, l]]]

    These are the respective probabilities:
    [[m, n], [o, p], [q, r]]
    """
    for batch_i, batch in enumerate(predictions):
        for mode_i, mode in enumerate(batch):
            for point_i, point in enumerate(mode):
                for p in point:
                    assert p in fc_out[batch_i, point_i, :]

