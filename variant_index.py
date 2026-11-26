from src.variants_vector_index.vector_loader import VectorLoader
import numpy as np
import time

vec_loader = VectorLoader(
    dataset="clinvar_CLNSIG_hyena-tiny",
    checkpoint="Run-GFM/luxnk59q/checkpoints/epoch=99-step=431100.ckpt",
)

query_vector = vec_loader.vectors[1]
query_vector_label = vec_loader.labels[1]
start_time = time.time()
distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=20)
end_time = time.time()
query_time = end_time - start_time

print(f"Query vector label: {query_vector_label}")
print(f"Distances: {distances}")
print(f"Result labels: {result_labels}")
print("========================")
print(f"Query vector size: {query_vector.shape}")
print(f"Query time: {query_time} seconds")
