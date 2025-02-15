# Routing


## Cluster Generation
1. Take a set of queries
2. Convert each query into a vector using a pre-trained embedding model
3. Cluster the queries into different groups based on the similarity of their vectors
4. For each cluster, select the query that is most representative of the cluster or convert the cluster into a single embedding vector

## Routing
1. Take a set of queries
2. Convert the queries into a vector using the same pre-trained embedding model
3. For each set of similar queries, convert the cluster into a single embedding vector
4. Return the query that is most representative of the cluster or the cluster embedding vector

