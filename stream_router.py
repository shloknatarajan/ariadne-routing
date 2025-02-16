import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import os
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors

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
    
    def __init__(self, agents, embedding_dim=64, learning_rate=0.01, min_samples=5):
        """
        agents: list of agent names.
        alpha: threshold for clustering (no longer used for update logic here).
        embedding_dim: dimension for both prompt projections and agent embeddings.
        learning_rate: learning rate used for updating embeddings.
        min_samples: minimum number of samples needed before training a cluster embedding.
        """
        self.agents = agents  # list of agent names
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.min_samples = min_samples
        self.cap = 10

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
        Performs multiple epochs of gradient updates on the cluster and agent embeddings using contrastive loss.
        
        Args:
            cluster_embedding: The cluster embedding Parameter to update
            agents_in_cluster: List of agent names involved in the update
            samples: List of (prompt_emb, agent) tuples to use for the update
        """
        params = [cluster_embedding] + [self.agent_embeddings[agent] for agent in agents_in_cluster]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        
        num_epochs = 8
        temperature = 0.1
        margin = 0.5  # Margin for contrastive loss
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Compute contrastive loss using samples in cluster
            for prompt_emb, positive_agent in samples:
                # Positive pair: cluster embedding and positive agent embedding
                positive_dist = 1 - F.cosine_similarity(
                    cluster_embedding.unsqueeze(0),
                    self.agent_embeddings[positive_agent].unsqueeze(0)
                )
                
                # Negative pairs: cluster embedding and other agent embeddings
                negative_loss = 0.0
                num_negatives = 0
                for agent_id in agents_in_cluster:
                    if agent_id != positive_agent:
                        negative_dist = 1 - F.cosine_similarity(
                            cluster_embedding.unsqueeze(0),
                            self.agent_embeddings[agent_id].unsqueeze(0)
                        )
                        # Hinge loss with margin
                        negative_loss += torch.max(torch.zeros_like(negative_dist),
                                                 margin - negative_dist)
                        num_negatives += 1
                
                if num_negatives > 0:
                    negative_loss = negative_loss / num_negatives
                    
                # Total loss combines positive and negative terms
                loss = positive_dist + negative_loss
                total_loss += loss

            # Average loss over all samples
            total_loss = total_loss / len(samples)
            total_loss.backward()
            optimizer.step()
            
            # Normalize embeddings
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
        Processes an update by embedding the prompt and performing a KNN search among the stored 
        prompt embeddings of clusters. With probability 0.2, the prompt is added to its nearest cluster.
        Otherwise, a new cluster is created. Clusters are capped at self.cap
         samples.
        """
        # Compute the embedding of the new prompt
        prompt_emb = self._compute_prompt_embedding(prompt)

        # Build candidate list from existing clusters
        candidates = []
        for cluster in self.clusters:
            candidates.append({
                "type": "cluster", 
                "object": cluster,
                "embedding": cluster["center"]
            })

        if not candidates:
            # Create first cluster if none exist
            new_cluster = {
                "data": [(prompt_emb, agent)],
                "embedding": None,
                "center": prompt_emb,
                "count": 1
            }
            self.clusters.append(new_cluster)
            return

        # Prepare candidate embeddings matrix for KNN search
        candidate_matrix = []
        for cand in candidates:
            emb = cand["embedding"]
            if isinstance(emb, torch.Tensor):
                emb_np = emb.detach().cpu().numpy()
            else:
                emb_np = np.array(emb)
            candidate_matrix.append(emb_np)
        candidate_matrix = np.stack(candidate_matrix)

        # Find nearest cluster
        nbrs = NearestNeighbors(n_neighbors=1).fit(candidate_matrix)
        query_vector = prompt_emb.detach().cpu().numpy().reshape(1, -1)
        distances, indices = nbrs.kneighbors(query_vector)
        
        nearest_cluster = candidates[indices[0][0]]["object"]

        # Create new cluster if nearest is full or with 10% probability
        if len(nearest_cluster["data"]) >= self.cap:# or np.random.random() < 0.1:
            # Create new cluster
            new_cluster = {
                "data": [(prompt_emb, agent)],
                "embedding": None,
                "center": prompt_emb,
                "count": 1,
            }
            self.clusters.append(new_cluster)
        else:
            # Add prompt to nearest cluster
            nearest_cluster["data"].append((prompt_emb, agent))
            nearest_cluster["count"] += 1
            
            # Update running average for cluster center
            nearest_cluster["center"] = (nearest_cluster["center"] * (1 - 1/nearest_cluster["count"]) +
                                       prompt_emb * (1/nearest_cluster["count"]))
            
            if len(nearest_cluster["data"]) == self.min_samples:
                self._training_update(nearest_cluster)
            elif len(nearest_cluster["data"]) > self.min_samples:
                self._training_update(nearest_cluster, new_data=(prompt_emb, agent))

    def inference(self, prompt: str) -> str:
        """
        Given a prompt, embeds it and checks for a trained cluster within alpha.
        
        Uses nearest neighbors search to find closest cluster center. If within alpha threshold,
        returns agent prediction from that cluster. Otherwise falls back to default.
        """
        # Embed the prompt
        prompt_emb = self._compute_prompt_embedding(prompt)
        
        # If no clusters exist yet, return default agent
        if not self.clusters:
            return self.agents[0]
            
        # Prepare candidate embeddings matrix for KNN search
        candidate_matrix = []
        for cluster in self.clusters:
            emb = cluster["center"]
            if isinstance(emb, torch.Tensor):
                emb_np = emb.detach().cpu().numpy()
            else:
                emb_np = np.array(emb)
            candidate_matrix.append(emb_np)
        candidate_matrix = np.stack(candidate_matrix)

        # Find nearest cluster
        nbrs = NearestNeighbors(n_neighbors=1).fit(candidate_matrix)
        query_vector = prompt_emb.detach().cpu().numpy().reshape(1, -1)
        distances, indices = nbrs.kneighbors(query_vector)
        
        nearest_cluster = self.clusters[indices[0][0]]
        
        if nearest_cluster["embedding"] is not None:
            # Use nearest neighbor to find closest agent embedding
            #using an embedding rather than a count allows us to account for new agents or removed agents
            agent_embeddings = torch.stack([self.agent_embeddings[a] for a in self.agents])
            distances = torch.norm(agent_embeddings - nearest_cluster["embedding"], dim=1)
            predicted_agent = self.agents[torch.argmin(distances).item()]
        else:
            # Get the most common agent in this cluster
            agent_counts = {}
            for _, agent in nearest_cluster["data"]:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            predicted_agent = max(agent_counts.items(), key=lambda x: x[1])[0]
        
        return predicted_agent
    
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

    def add_agent(self, agent_name: str, example_prompts: list[str] = None):
        """
        Adds a new agent to the router with optional example prompts to create an initial cluster.
        
        Args:
            agent_name: str - Name of the new agent to add
            example_prompts: Optional[List[str]] - List of example prompts for this agent
        """
        # Check if agent already exists
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' already exists")
            
        # Add to agents list
        self.agents.append(agent_name)
        
        # Create new embedding for agent
        self.agent_embeddings[agent_name] = nn.Parameter(
            torch.randn(self.embedding_dim), 
            requires_grad=True
        )
        
        # If example prompts provided, create a new cluster
        if example_prompts and len(example_prompts) > 0:
            # Compute embeddings for all examples
            prompt_embeddings = [self._compute_prompt_embedding(p) for p in example_prompts]
            
            # Create new cluster with examples
            cluster_data = [(emb, agent_name) for emb in prompt_embeddings]
            
            # Calculate center as average of embeddings
            center = torch.stack(prompt_embeddings).mean(dim=0)
            
            new_cluster = {
                "data": cluster_data,
                "embedding": None,
                "center": center,
                "count": len(example_prompts)
            }
            
            self.clusters.append(new_cluster)
            
            # If we have enough samples, train the cluster embedding
            if len(example_prompts) >= self.min_samples:
                self._training_update(new_cluster)

if __name__ == "__main__":
    # Import training updates and test prompts from the external file.
    from ideation import TRAIN_UPDATES, TEST_PROMPTS

    # Create a router instance with desired parameters.
    agents = ["HR Agent", "Code Generation Agent","Web Search Agent", "Customer Service Agent", "Database Agent"]
    router = StreamRouter(agents,  embedding_dim=64, learning_rate=0.1)
    
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
