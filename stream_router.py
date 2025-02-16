import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import os
from openai import OpenAI

os.environ['TRUST_REMOTE_CODE'] = '1'

class StreamRouter:
    """
    A streaming router that performs dynamic clustering with a threshold (alpha) and uses
    an oracle to determine the correct agent.
    
    For each prompt:
      - Its embedding is computed and projected into an internal space.
      - If the prompt's embedding is within a threshold (alpha) of an existing cluster's centroid (or candidate center), 
        the prompt is assigned to that cluster.
      - Otherwise, a new cluster is created for the prompt.
    
    For clusters with fewer than min_samples examples, the oracle_router() function is called to yield the correct agent,
    and that information is stored into the cluster.
    
    Once a cluster reaches min_samples elements, the cluster embedding is established using the min_samples elements (or updated via a streaming rule).
    """
    
    def __init__(self, agents, alpha=0.5, embedding_dim=64, learning_rate=0.01, min_samples=5, model_name = 'all-distilroberta-v1'):
        """
        agents: list of agent names.
        alpha: threshold for clustering.
        embedding_dim: dimension for both prompt projections and agent embeddings.
        learning_rate: learning rate used for updating embeddings.
        min_samples: minimum number of samples needed before training a cluster embedding.
        """
        self.alpha = alpha
        self.agents = agents  # list of agent names
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.min_samples = min_samples

        # Initialize OpenAI client and embedding model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embed_model = lambda text: self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        # Initialize agent embeddings as learnable parameters.
        # They are stored in a dictionary mapping agent->nn.Parameter.
        self.agent_embeddings = {
            agent: nn.Parameter(torch.randn(embedding_dim), requires_grad=True)
            for agent in agents
        }

        # This list will store clusters.
        # Each cluster is a dict with:
        #   "data": a list of tuples (prompt_embedding, agent)
        #   "embedding": a torch.nn.Parameter (learned after at least min_samples samples) or None.
        #   "center": an average embedding of the cluster.
        #   "count": the number of samples in the cluster.
        self.clusters = []
    
    def _compute_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """
        Uses the sentence transformer model to encode the prompt and returns a torch tensor.
        """
        embedding_np = self.embed_model(prompt)
        return torch.tensor(embedding_np, dtype=torch.float32)
    
    
    def _distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Computes the Euclidean distance between two vectors.
        """
        return torch.norm(vec1 - vec2).item()
    
    def _similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Computes the cosine similarity between two vectors.
        """
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def _perform_update(self, cluster_embedding, agents_in_cluster, samples):
        """
        Performs a single gradient update step on the cluster and agent embeddings using contrastive loss.
        
        Args:
            cluster_embedding: The cluster embedding Parameter to update
            agents_in_cluster: List of agent names involved in the update
            samples: List of (prompt_emb, agent) tuples to use for the update
        """
        params = [cluster_embedding] #+ [self.agent_embeddings[agent] for agent in agents_in_cluster]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
        margin = 0.1  # Margin for contrastive loss
        
        for prompt_emb, positive_agent in samples:
            # Get positive pair similarity
            pos_sim = F.cosine_similarity(
                cluster_embedding.unsqueeze(0),
                self.agent_embeddings[positive_agent].unsqueeze(0)
            )
            
            # Compute loss against negative pairs
            for agent_id in agents_in_cluster:
                if agent_id != positive_agent:
                    neg_sim = F.cosine_similarity(
                        cluster_embedding.unsqueeze(0),
                        self.agent_embeddings[agent_id].unsqueeze(0)
                    )
                    # Contrastive loss: minimize distance to positive, maximize to negative
                    loss = loss + torch.max(torch.tensor(0.0, requires_grad=True), margin - (pos_sim - neg_sim))
         
        loss = loss / len(samples)
        loss.backward()
        optimizer.step()
        
        # Normalize embeddings after update
        with torch.no_grad():
            cluster_embedding.data = cluster_embedding.data / (cluster_embedding.data.norm(p=2) + 1e-8)
            for agent in agents_in_cluster:
                self.agent_embeddings[agent].data = self.agent_embeddings[agent].data / (self.agent_embeddings[agent].data.norm(p=2) + 1e-8)

    def _training_update(self, cluster: dict, new_data=None):
        """
        Trains (or updates) the cluster embedding and the corresponding agent embedding(s).
        
        If the cluster has not yet been initiated (i.e. "embedding" is None),
        the cluster embedding is created as the average of stored prompt embeddings and updated
        using all stored (prompt, agent) pairs.
        
        If the cluster is already initiated and new_data (a single tuple or list of tuples)
        is provided, a single gradient update is performed using the new sample(s).
        """
        if cluster["embedding"] is None:
            # Initiate the cluster embedding as the average of stored prompt embeddings.
            embeddings = [sample[0] for sample in cluster["data"]]
            cluster["embedding"] = nn.Parameter(torch.normal(0,1,size=(self.embedding_dim,)))

            # Collect names of agents in this cluster.
            agents_in_cluster = list({sample[1] for sample in cluster["data"]})
            self._perform_update(cluster["embedding"], agents_in_cluster, cluster["data"])
        else:
            # Already initiated; update using the new sample(s).
            if new_data is None:
                return
            # Ensure new_data is in list form.
            if not isinstance(new_data, list):
                new_data = [new_data]
            agents_in_update = list({sample[1] for sample in new_data})
            self._perform_update(cluster["embedding"], agents_in_update, new_data)

    def update(self, prompt: str, agent: str):
        """
        Processes an update by embedding the prompt and adding the (prompt, agent) pair to an appropriate cluster.
        
        If the embedded prompt is within alpha of an existing cluster's candidate vector,
        the pair is added to that cluster.
          - When the cluster's storage reaches min_samples examples, an initial training update is applied.
          - For subsequent additions to the cluster (size > min_samples), a single-step update is applied.
          
        If no cluster is within alpha, a new cluster is created.
        """
        prompt_emb = self._compute_prompt_embedding(prompt)
        best_cluster = None
        best_dist = None

        # Try to find an existing cluster (compare with candidate vector)
        for cluster in self.clusters:
            d = self._distance(prompt_emb, cluster["center"])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_cluster = cluster

        if best_cluster is not None and best_dist < self.alpha:
            # Add new sample to the best cluster.
            best_cluster["data"].append((prompt_emb, agent))
            best_cluster["count"] += 1
            best_cluster["center"] = best_cluster["center"] * (1 - 1/best_cluster['count']) + prompt_emb * 1/best_cluster['count']

            if len(best_cluster["data"]) == self.min_samples:
                # Initiate and train the cluster embedding using stored data.
                self._training_update(best_cluster)
            elif len(best_cluster["data"]) > self.min_samples:
                # Perform a single training update step using the new sample.
                self._training_update(best_cluster, new_data=(prompt_emb, agent))
        else:
            # No suitable cluster found; create a new one.
            new_cluster = {
                "data": [(prompt_emb, agent)],
                "embedding": None,
                "center": prompt_emb,
                "count": 1
            }
            self.clusters.append(new_cluster)

    def inference(self, prompt: str) -> str:
        """
        Given a prompt, embeds it and checks for a trained cluster within alpha.
        
        If found, the cluster embedding is used to compute logits with all agent embeddings,
        softmax is applied, and the agent with the highest probability is returned.
        If no suitable cluster exists, the oracle_router is called.
        """
        prompt_emb = self._compute_prompt_embedding(prompt)
        best_cluster = None
        best_dist = None
        for cluster in self.clusters:
            d = self._distance(prompt_emb, cluster["center"])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_cluster = cluster

        if best_cluster is not None and best_dist < self.alpha and best_cluster["count"] > self.min_samples:
            # Compute dot products between the cluster embedding and each agent embedding.
            cluster_emb = best_cluster["embedding"]
            logits = []
            for agent in self.agents:
                agent_emb = self.agent_embeddings[agent]
                logits.append(torch.dot(cluster_emb, agent_emb))
            logits = torch.stack(logits)
            probs = F.softmax(logits, dim=0)
            pred_idx = torch.argmax(probs).item()
            return self.agents[pred_idx]
        else:
            print(f"Unable to find a cluster for prompt")
            return "welp"
    
    def debug_clusters(self):
        """
        Debug method to print current clusters and their contents.
        """
        print("\nClusters:")
        for cluster in self.clusters:
            print(f"\nCluster | Count: {len(cluster['data'])}")
            print("Prompts:")
            for prompt_emb, agent in cluster["data"]:
                print(f"  - {prompt_emb}")

if __name__ == "__main__":
    # Import training updates and test prompts from the external file.
    from ideation import TRAIN_UPDATES, TEST_PROMPTS

    # Create a router instance with desired parameters.
    agents = ["HR Agent", "Code Generation Agent","Web Search Agent", "Customer Service Agent", "Database Agent"]
    router = StreamRouter(agents, alpha=23, embedding_dim=64, learning_rate=0.1)
    
    # Process training updates (each prompt is paired with an agent, but we ignore the provided agent).
    for prompt, agent in TRAIN_UPDATES:
        router.update(prompt, agent)
    print(f"Total clusters formed after training update: {len(router.clusters)}")
    
    # Debug: Print cluster details.
    router.debug_clusters()
    
    # Execute test inferences using the imported test prompts.
    print("\nTest Inference Results:")
    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        predicted_agent = router.inference(prompt)
        print(f"\nTest Prompt {i}: \"{prompt}\"")
        print(f"Predicted Agent: {predicted_agent}")
