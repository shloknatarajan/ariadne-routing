# Routing
Problem: We want to be able to route a user's query to the best agent based on the query. This solves an increasingly difficult problem as the number of agents grows. This solution is inspired by social media recommendation systems to recommend agents based on the user's query. 

To get started, run `python app.py`

## Cluster Generation
1. Take a set of queries
2. Convert each query into a vector using a pre-trained embedding model
3. Cluster the queries into different groups based on the similarity of their vectors
4. For each cluster, select the query that is most representative of the cluster or convert the cluster into a single embedding vector

## Routing
1. Take a set of queries
2. Convert the queries into a vector using the same pre-trained embedding model
3. For each set of similar queries, convert the cluster into a single embedding vector
4. Return cluster embedding vector
5. Map cluster embedding vector to agent space
6. Map to agent embedding

