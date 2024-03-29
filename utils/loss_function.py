import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Idea for contrastive learning:
(1). Contrastive learning is primarily used to enhance query factor disentanglement, not necessarily to increase predict
-ion power. We assume the model should already be capable of achieving good prediction performance.
(2). The similarity between neurons will be calculated based on the spatial and temporal responses to the same stimuli, 
which in this case are noise and chirps.
(3). For contrastive learning, the stimuli are drawn from the same stimuli set, but from different neurons. The similar-
ity index is then calculated based on these responses.
(4). To increase the number of positive examples, we can search for different stimuli that elicit similar firing rates 
from the neuron. Selecting these stimuli from existing data might be a simpler approach.
"""

def dynamic_margin_triplet_loss(anchor, positive, negative, similarity_ap, similarity_an, base_margin=1.0, alpha=0.5):
    """
    For contrastive learning (self-supervised)
    Computes the triplet loss with dynamic margins based on similarity scores.

    Parameters:
    - anchor, positive, negative: embeddings for the anchor, positive, and negative samples.
    - similarity_ap: Tensor containing similarity scores between anchor and positive samples.
    - similarity_an: Tensor containing similarity scores between anchor and negative samples.
    - base_margin: The base margin value.
    - alpha: Scaling factor to adjust the influence of similarity scores on the margin.

    Returns:
    - loss: The computed triplet loss with dynamic margins.
    """

    # Ensure similarity scores are in a suitable range (e.g., [0, 1])
    similarity_ap = torch.sigmoid(similarity_ap)
    similarity_an = torch.sigmoid(similarity_an)

    # Calculate dynamic margin
    dynamic_margin = base_margin + alpha * (similarity_an - similarity_ap)

    # Calculate pairwise distance
    distance_ap = F.pairwise_distance(anchor, positive, p=2)
    distance_an = F.pairwise_distance(anchor, negative, p=2)

    # Compute triplet loss
    losses = F.relu(distance_ap - distance_an + dynamic_margin)
    return losses.mean()


def soft_contrastive_loss_pow2(embeddings1, embeddings2, similarity_scores, margin=1.0):
    """
    For contrastive learning (self-supervised)
    Computes a soft contrastive loss given pairs of embeddings and their similarity scores.

    Parameters:
    - embeddings1, embeddings2: Tensors of shape (batch_size, embedding_size) representing pairs of embeddings.
    - similarity_scores: Tensor of shape (batch_size,) containing continuous similarity scores for each pair.
    - margin: A margin term for dissimilar pairs.

    Returns:
    - loss: The computed soft contrastive loss.
    """

    # Calculate the Euclidean distance between pairs of embeddings
    distances = F.pairwise_distance(embeddings1, embeddings2, p=2)

    # Convert similarity scores to a scale that matches the distances
    similarity_scores_scaled = similarity_scores * margin

    # Compute the loss for similar and dissimilar pairs differently
    loss_similar = (1 - similarity_scores) * distances.pow(2)  # Loss for similar pairs
    loss_dissimilar = similarity_scores * F.relu(margin - distances).pow(2)  # Loss for dissimilar pairs

    # Combine the losses
    loss = loss_similar + loss_dissimilar
    return loss.mean()


def soft_contrastive_loss_log(embeddings_i, embeddings_j, similarity_scores):
    """
    Computes the soft contrastive loss for a pair of embeddings and a similarity score.

    Parameters:
    - embeddings_i: Tensor of shape (batch_size, embedding_size) representing embeddings for instance i.
    - embeddings_j: Tensor of shape (batch_size, embedding_size) representing embeddings for instance j.
    - similarity_scores: Tensor of shape (batch_size,) containing continuous similarity scores between 0 and 1 for each pair.

    Returns:
    - loss: The computed soft contrastive loss.
    """
    # Calculate the sigmoid of the dot product between the pairs of embeddings
    sim_prob = torch.sigmoid(torch.sum(embeddings_i * embeddings_j, dim=1))

    # Compute the binary cross-entropy loss
    loss = -similarity_scores * torch.log(sim_prob) - (1 - similarity_scores) * torch.log(1 - sim_prob)
    return loss.mean()


class CosineNegativePairLoss(nn.Module):
    def __init__(self, margin=0.1, temperature=0.1):
        """
        Initialize the loss function with a margin and a temperature parameter.

        :param margin: A scalar that defines the threshold for penalizing negative pairs
                       that are too close in terms of cosine distance.
        :param temperature: A scalar that scales the cosine distances, controlling
                            the separation in the embedding space.
        """
        super(CosineNegativePairLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings, anchors):
        """
        Compute the loss for each pair of negative examples in the batch based on scaled cosine distance.

        :param embeddings: Tensor of shape (batch_size, embedding_dim) containing the embeddings
                           of the samples. Assumes that each consecutive pair of samples in the batch
                           are considered a negative pair.
        :param anchors: Tensor of shape (batch_size, embedding_dim) containing the embeddings
                        of the corresponding anchors. Assumes that each embedding in the `embeddings`
                        tensor has a corresponding anchor in this tensor.
        """
        # Normalize the embeddings and anchors to have unit norm
        embeddings = F.normalize(embeddings, p=2, dim=1)
        anchors = F.normalize(anchors, p=2, dim=1)

        # Calculate cosine similarity for all possible pairs
        cosine_sim = F.cosine_similarity(embeddings, anchors, dim=1)

        # Convert cosine similarity to cosine distance
        cosine_dist = 1 - cosine_sim

        # Apply temperature scaling
        scaled_cosine_dist = cosine_dist / self.temperature

        # Calculate loss based on the margin
        losses = F.softplus(self.margin - scaled_cosine_dist)

        # Return the mean loss over all negative pairs
        return losses


# Define the mapping of loss function names to their PyTorch implementations
loss_functions = {
    'MSE': nn.MSELoss(),
    'Poisson': nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean'),
    # Add additional loss functions here as needed
}