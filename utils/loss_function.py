import torch
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


def soft_contrastive_loss(embeddings1, embeddings2, similarity_scores, margin=1.0):
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
