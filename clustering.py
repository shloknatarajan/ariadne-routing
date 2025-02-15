from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Dict
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def embed_sentence(sentence: str, model_name: str = "bert-base-nli-mean-tokens"):
    """
    Embeds a sentence using a BERT-based model from Hugging Face's Sentence-Transformers.
    
    :param sentence: The input sentence to be embedded.
    :param model_name: The name of the pre-trained model to use (default: "bert-base-nli-mean-tokens").
    :return: A vector embedding (numpy array) of the sentence.
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode(sentence)
    return embedding

def cluster_sentences(sentences: List[str], n_clusters: int = 5, model_name: str = "bert-base-nli-mean-tokens", show_graph: bool = False) -> Dict[int, List[str]]:
    """
    Clusters sentences using their embeddings and K-Means clustering.
    Automatically determines optimal number of clusters using elbow method.
    
    :param sentences: List of sentences to cluster
    :param n_clusters: Maximum number of clusters to create (default: 2)
    :param model_name: The name of the pre-trained model to use
    :param show_graph: If True, displays a 2D visualization of the clusters (default: False)
    :return: Dictionary mapping cluster IDs to lists of sentences
    """
    # Create embeddings for all sentences
    embeddings = np.array([embed_sentence(sent, model_name) for sent in sentences])
    
    # Find optimal number of clusters using elbow method
    inertias = []
    K = range(1, min(n_clusters + 1, len(sentences)))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    
    # Calculate the rate of change in inertia
    elbow_point = 1  # Default to 1 cluster if no clear elbow
    if len(K) > 2:
        changes = np.diff(inertias)
        # Find point where the rate of improvement slows down significantly
        threshold = np.mean(np.abs(changes)) * 0.5  # Adjust threshold as needed
        for i, change in enumerate(changes):
            if abs(change) < threshold:
                elbow_point = i + 1
                break
    
    optimal_clusters = min(elbow_point, n_clusters)
    
    # Perform K-means clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # If show_graph is True, create and display the visualization
    if show_graph:
        # Reduce dimensionality to 2D using t-SNE
        perplexity = min(30, len(sentences) - 1)  # Adaptive perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        for i in range(optimal_clusters):
            # Plot points for each cluster with different colors
            mask = cluster_labels == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Cluster {i}')
        
        plt.title(f'Sentence Clusters Visualization (Optimal clusters: {optimal_clusters})')
        plt.legend()
        plt.show()
    
    # Group sentences by cluster
    clusters = {i: [] for i in range(optimal_clusters)}
    i=0
    for sentence, label in zip(sentences, cluster_labels):
        clusters[label].append([sentence,embeddings[i]])
        i+=1

    for label in range(optimal_clusters):
        mask = cluster_labels == label
        cluster_embeddings = embeddings[mask]
        avg_embedding = np.mean(cluster_embeddings, axis=0)
        clusters[label].append(avg_embedding)
    return clusters

# Example usage
sentence = "This is a sample sentence."
vector = embed_sentence(sentence)
print(vector.shape)  # Output: (768,)

if __name__ == "__main__":
    test_sentences = [
        "The cat sits on the mat",
        "The cat sits on the bed",
        "The cat sits on the table",
        "Neural networks process data",
        "Neural networks process lots of data",
        "Neural networks process information",
        "The dog runs on the mat"
    ]
    
    clusters = cluster_sentences(test_sentences, show_graph=False)
    for cluster_id in clusters:
        print(clusters[cluster_id][-1])
