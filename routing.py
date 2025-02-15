from clustering import main
import numpy as np

clusters = main()
agents = [i for i in range(3)]

def initialize(clusters, agents):
    cluster_vectors = {i:clusters[label][-1] for i,label in enumerate(clusters)}
    # Create a dictionary with random Gaussian vectors for each agent
    agent_vectors = {i:np.random.normal(0, 1, 768) for i in range(len(agents))}
    
    return cluster_vectors, agent_vectors


def learn_embeddings(cluster_vectors, agent_vectors, groud_truth):
    pass

