import torch

def minADE(predictions, probabilities, gt, k=-1):
    distances = torch.norm(predictions - gt.unsqueeze(1), dim=-1)
    ret = torch.empty((predictions.shape[0], predictions.shape[1]))

    sorted_idx = torch.argsort(probabilities, descending=True)
    top_k_idx = sorted_idx[:, :k]

    for sample in range(len(predictions)):
        l2_distances = distances[sample][top_k_idx[sample]]
        ret = l2_distances.sum()

    return distances

# predictions is size (batch_size, num_modes, predictions_per_mode, 2) aka (16, 5, 12, 2)
# probabilities is size (batch_size, num_modes) aka (16, 5)
# ground_truth is size (batch_size, predictions_per_mode, 2) aka (16, 12, 2)

if __name__ == "__main__":
    predictions = torch.ones(16, 5, 12, 2)
    probabilities = torch.ones(16, 5)
    ground_truth = torch.ones(16, 12, 2)

    print(minADE(predictions, probabilities, ground_truth, k=6).any())
    
