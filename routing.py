import torch
import torch.nn.functional as F
from clustering import main



def initialize(clusters, agents):
    cluster_vectors = {i: torch.tensor(clusters[label][-1], dtype=torch.float32, requires_grad=True) 
                      for i, label in enumerate(clusters)}
    # Create a dictionary with random Gaussian vectors for each agent
    agent_vectors = {i: torch.randn(768, requires_grad=True) 
                    for i in range(len(agents))}
    
    return cluster_vectors, agent_vectors

def get_distribution(cluster_vector, agent_vectors, temperature=0.1):
    # Calculate cosine similarities
    similarities = torch.stack([
        F.cosine_similarity(cluster_vector.unsqueeze(0), 
                          agent_vec.unsqueeze(0)) 
        for agent_vec in agent_vectors.values()
    ])
    # Convert to logits/probabilities via softmax
    probs = F.softmax(similarities / temperature, dim=0)
    return probs

def kl_divergence(p, q):
    # Using PyTorch's built-in KL divergence
    return F.kl_div(q.log(), p, reduction='sum')

def learn_embeddings(cluster_vectors, agent_vectors, ground_truth, epochs=100, learning_rate=0.01, temperature=0.1, alpha=0.3):
    """
    ground_truth: dict mapping cluster_id to (agent_indices, probabilities)
                 where probabilities are the target distribution over agents
    Args:
        alpha: Regularization strength for cluster embedding deviation
    """
    # Store original cluster vectors for regularization
    original_cluster_vectors = {
        k: v.clone().detach() 
        for k, v in cluster_vectors.items()
    }
    
    n_agents = len(agent_vectors)
    optimizer = torch.optim.Adam(
        list(cluster_vectors.values()) + list(agent_vectors.values()), 
        lr=learning_rate
    )
    
    for epoch in range(epochs):
        total_loss = 0
        
        # For each cluster
        for cluster_id, cluster_vec in cluster_vectors.items():
            optimizer.zero_grad()
            
            # Get ground truth distribution
            true_agents, true_probs = ground_truth[cluster_id]
            
            # Create full probability distribution (zeros for non-top-4 agents)
            true_distribution = torch.zeros(n_agents)
            for agent_idx, prob in zip(true_agents, true_probs):
                true_distribution[agent_idx] = prob
            
            # Get current predicted distribution
            pred_distribution = get_distribution(cluster_vec, agent_vectors, temperature)
            
            # Calculate KL divergence loss
            kl_loss = kl_divergence(true_distribution, pred_distribution)
            
            # Add regularization term for cluster embedding
            cluster_reg = F.mse_loss(cluster_vec, original_cluster_vectors[cluster_id])
            loss = kl_loss + alpha * cluster_reg
            
            total_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Normalize vectors after optimization
            with torch.no_grad():
                for vec in agent_vectors.values():
                    vec.div_(vec.norm())
                cluster_vec.div_(cluster_vec.norm())
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss/len(cluster_vectors):.4f}")
    
    return cluster_vectors, agent_vectors

def inference(cluster_vector, agent_vectors, temperature=0.1):
    """
    Returns the probability distribution over agents for a given cluster
    """
    with torch.no_grad():
        probs = get_distribution(cluster_vector, agent_vectors, temerature=temperature)
    return probs.numpy()


clusters = main()
agents = [i for i in range(3)]
# Example usage
cluster_vectors, agent_vectors = initialize(clusters, agents)
trained_cluster_vectors, trained_agent_vectors = learn_embeddings(
    cluster_vectors, 
    agent_vectors,  
    ground_truth
)

# For inference
probs = inference(trained_cluster_vectors[0], trained_agent_vectors)

