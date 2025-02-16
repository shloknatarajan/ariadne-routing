import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import os
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from langchain.docstore.document import Document
from langchain_iris import IRISVector

# A custom embedding function for our clusters.
# Given a document whose metadata contains the cluster index, return the corresponding cluster center.
class ClusterCenterEmbeddings:
    def __init__(self, router):
        self.router = router

    def embed(self, text: str, metadata: dict = None) -> np.ndarray:
        """
        Instead of computing an embedding from the text, we look up the cluster center.
        We assume that metadata contains the cluster index.
        """
        cluster_index = metadata.get("cluster_index")
        center = self.router.clusters[cluster_index]["center"]
        # Make sure to return a NumPy array.
        return center.detach().cpu().numpy()

# In your StreamRouter __init__, add (or create on first use) an IRISVector store for clusters.
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
        # (existing initialization code)
        self.agents = agents
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.min_samples = min_samples
        self.cap = 10

        # Initialize OpenAI client and embedding model as before.
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embed_model = lambda text: self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        # Initialize agent embeddings as learnable parameters.
        self.agent_embeddings = {
            agent: torch.nn.Parameter(torch.randn(embedding_dim), requires_grad=True)
            for agent in agents
        }

        self.clusters = []  # list of clusters

        # IRIS connection details; adjust these as needed.
        username = 'demo'
        password = 'demo'
        hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
        port = '1972'
        namespace = 'USER'
        self.CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

        # We will lazily initialize the IRISVector store for clusters when needed.
        self.cluster_vector_store = None
        self.cluster_collection_name = "stream_router_clusters"

    def _update_cluster_vector_store(self):
        """
        Re-create (or update) the IRISVector store with the current cluster centers.
        We convert each cluster into a Document whose text is a dummy value (e.g., "cluster")
        and whose metadata contains the cluster index.
        """
        from langchain.docstore.document import Document
        
        docs = []
        for i, cluster in enumerate(self.clusters):
            docs.append(Document(page_content="cluster", metadata={"cluster_index": i}))
        
        # Create a custom embedding function that looks up the correct cluster center.
        cluster_embedding_func = ClusterCenterEmbeddings(self)
        
        # Create (or re-create) the IRISVector store for clusters.
        self.cluster_vector_store = IRISVector.from_documents(
            documents=docs,
            embedding=cluster_embedding_func,  # Our custom function
            collection_name=self.cluster_collection_name,
            connection_string=self.CONNECTION_STRING,
        )

    def inference(self, prompt: str) -> str:
        """
        Given a prompt, embed it and use IRISVector Search to find the nearest cluster.
        Then, based on the nearest cluster, select an agent.
        """
        # Embed the prompt as before.
        prompt_emb = self._compute_prompt_embedding(prompt)
        
        if not self.clusters:
            return self.agents[0]  # default if no clusters yet
        
        # Make sure our cluster vector store is up-to-date.
        self._update_cluster_vector_store()
        
        # Because IRISVector expects a text query and will use the embedding function provided
        # (which in our case ignores the text and uses the stored cluster center), we pass the prompt.
        # (If desired, you could also write a custom method that accepts a vector directly.)
        docs_with_score = self.cluster_vector_store.similarity_search_with_score(prompt, k=1)
        
        # Retrieve the cluster index from the returned document.
        nearest_cluster_index = docs_with_score[0][0].metadata["cluster_index"]
        nearest_cluster = self.clusters[nearest_cluster_index]
        
        # (Now do the rest as before: if the cluster is trained, choose the nearest agent embedding.)
        if nearest_cluster["embedding"] is not None:
            # Instead of computing the distances manually, you could similarly maintain an IRISVector store for agents.
            # For brevity, we keep this step in memory.
            agent_embs = torch.stack([self.agent_embeddings[a] for a in self.agents])
            distances = torch.norm(agent_embs - nearest_cluster["embedding"], dim=1)
            predicted_agent = self.agents[torch.argmin(distances).item()]
        else:
            # Use most common agent if no trained cluster embedding.
            agent_counts = {}
            for _, agent in nearest_cluster["data"]:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            predicted_agent = max(agent_counts.items(), key=lambda x: x[1])[0]
        
        return predicted_agent
    

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