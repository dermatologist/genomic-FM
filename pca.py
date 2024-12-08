import numpy as np
import pickle
from sklearn.decomposition import PCA

with open ('data/my_data.pkl', 'rb') as fp:
    data = pickle.load(fp)

chunk = data

seq1s = [np.array(d[0][0]) for d in chunk]

# seq1s is a list of np arrays with three dimensions
# Let's pad the second dimension to a length of 512
for i in range(len(seq1s)):
    # pad second dimension of array with 3 dimensions
    # seq1s[i] = np.pad(seq1s[i], ((0, 512 - seq1s[i].shape[0]), (0, 0)), 'constant', constant_values=0)
    seq1s[i] = np.pad(seq1s[i], ((0, 0), (0, 512 - seq1s[i].shape[1]), (0, 0)), 'constant', constant_values=0)
    print(seq1s[i].shape)

seq1s = np.array(seq1s)

print(f"seq1s shape: {seq1s.shape}")
reshaped_data = seq1s.reshape(-1, seq1s.shape[-1])
print(f"reshaped_data shape: {reshaped_data.shape}")

pca = PCA(n_components=4)
pca.fit(reshaped_data)
print(pca.explained_variance_ratio_)

pca_transformed_data = pca.transform(reshaped_data)
pca_transformed_data = pca_transformed_data.reshape(seq1s.shape[:-1] + (-1,))

pca_transformed_data = pca_transformed_data.squeeze() # Reshape back to (num_samples, 1024, n_components)

print(f"pca_transformed_data shape: {pca_transformed_data.shape}")