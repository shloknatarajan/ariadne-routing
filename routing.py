from clustering import main
import numpy as np

clusters = main()


def embed(clusters, agents):
    cluster_vectors = {i:clusters[label][-1] for i,label in enumerate(clusters)}
    # Create a dictionary with random Gaussian vectors for each agent
    agent_vectors = {i:np.random.normal(0, 1, 768) for i in range(len(agents))}
    
    return cluster_vectors, agent_vectors

