import torch
from torch import _dynamo as torchdynamo

torchdynamo.optimize()
def minADE(predictions, probabilities, gt, k=-1):
    distances = torch.norm(predictions - gt.unsqueeze(1), dim=-1)
    ret = torch.empty((predictions.shape[0],))

    sorted_idx = torch.argsort(probabilities, descending=True)
    top_k_idx = sorted_idx[:, :k]

    for sample in range(len(predictions)):
        l2_distances = distances[sample][top_k_idx[sample]]
        ret[sample] = torch.min(l2_distances.sum(dim=-1))

    return ret

torchdynamo.optimize()
def minFDE(predictions, probabilities, gt, k=-1):
    distances = torch.norm(predictions - gt.unsqueeze(1), dim=-1)
    sorted_idx = torch.argsort(probabilities, descending=True)

    final_points = distances[:, :, -1]
    minFDE = torch.min(final_points, dim=1).values

    return minFDE

# predictions is size (batch_size, num_modes, predictions_per_mode, 2) aka (16, 5, 12, 2)
# probabilities is size (batch_size, num_modes) aka (16, 5)
# ground_truth is size (batch_size, predictions_per_mode, 2) aka (16, 12, 2)

if __name__ == "__main__":
    predictions = torch.ones(16, 5, 12, 2)
    probabilities = torch.ones(16, 5)
    ground_truth = torch.ones(16, 12, 2)

    assert minADE(predictions, probabilities, ground_truth, k=6).any() == False
    assert minFDE(predictions, probabilities, ground_truth, k=6).any() == False

    a = torch.tensor([
        [1, 2],
        [3, 4],
    ], dtype=torch.float32)

    predictions = torch.stack((a, a+1, a+2, a+3)).unsqueeze(0)
    probabilities = torch.tensor([0.40, 0.35, 0.20, 0.05]).unsqueeze(0)
    ground_truth = a.unsqueeze(0)

    assert minADE(predictions, probabilities, ground_truth, k=2).item() == 0.
    assert minADE(predictions + 1, probabilities, ground_truth, k=2).item() - 2 * 8**0.5 < 1e-5

    assert minFDE(predictions, probabilities, ground_truth, k=2).item() == 0.
    assert minFDE(predictions + 1, probabilities, ground_truth, k=2).item() - 2 * 8**0.5 < 1e-5
    
