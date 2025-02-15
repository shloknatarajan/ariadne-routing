from clustering import main
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

clusters = main()
agents = [i for i in range(3)]

def initialize(clusters, agents):
    cluster_vectors = {i:clusters[label][-1] for i,label in enumerate(clusters)}
    # Create a dictionary with random Gaussian vectors for each agent
    agent_vectors = {i:np.random.normal(0, 1, 768) for i in range(len(agents))}
    
    return cluster_vectors, agent_vectors

def get_distribution(cluster_vector, agent_vectors):
    # Calculate cosine similarities
    similarities = np.array([
        cosine_similarity(cluster_vector, agent_vec) 
        for agent_vec in agent_vectors.values()
    ])
    # Convert to logits/probabilities via softmax
    logits = similarities / 0.1  # Temperature parameter to control softmax sharpness
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

def kl_divergence(p, q):
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p/q))

def learn_embeddings(cluster_vectors, agent_vectors, ground_truth, epochs=100, learning_rate=0.01):
    """
    ground_truth: dict mapping cluster_id to (agent_indices, probabilities)
                 where probabilities are the target distribution over agents
    """
    n_agents = len(agent_vectors)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # For each cluster
        for cluster_id, cluster_vec in cluster_vectors.items():
            # Get ground truth distribution
            true_agents, true_probs = ground_truth[cluster_id]
            
            # Create full probability distribution (zeros for non-top-4 agents)
            true_distribution = np.zeros(n_agents)
            for agent_idx, prob in zip(true_agents, true_probs):
                true_distribution[agent_idx] = prob
            
            # Get current predicted distribution
            pred_distribution = get_distribution(cluster_vec, agent_vectors)
            
            # Calculate KL divergence loss
            loss = kl_divergence(true_distribution, pred_distribution)
            total_loss += loss
            
            # Gradient update
            # Here we update both cluster and agent vectors to minimize KL divergence
            # The gradient calculation is approximated through the softmax
            for agent_id, agent_vec in agent_vectors.items():
                error = pred_distribution[agent_id] - true_distribution[agent_id]
                grad = error * cluster_vec
                agent_vectors[agent_id] -= learning_rate * grad
                # Normalize to prevent exploding values
                agent_vectors[agent_id] /= np.linalg.norm(agent_vectors[agent_id])
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average KL Loss: {total_loss/len(cluster_vectors):.4f}")
    
    return cluster_vectors, agent_vectors

def inference(cluster_vector, agent_vectors):
    """
    Returns the probability distribution over agents for a given cluster
    """
    probs = get_distribution(cluster_vector, agent_vectors)
    return probs

# ground_truth format example:
# {cluster_id: ([agent1, agent2, agent3, agent4], [0.4, 0.3, 0.2, 0.1])}
cluster_vectors, agent_vectors = initialize(clusters, agents)
trained_cluster_vectors, trained_agent_vectors = learn_embeddings(
    cluster_vectors, 
    agent_vectors, 
    ground_truth
)

# For inference
probs = inference(trained_cluster_vectors, trained_agent_vectors)

