import numpy as np

# Load the embeddings from the first .npz file
data1 = np.load('mes.npz')
embeddings1 = data1['arr_0']
names1 = data1['arr_1']

# Load the embeddings from the second .npz file
data2 = np.load('ro.npz')
embeddings2 = data2['arr_0']
names2 = data2['arr_1']

# Load the embeddings from the third .npz file
data3 = np.load('ss.npz')
embeddings3 = data3['arr_0']
names3 = data3['arr_1']

# Combine the embeddings and names
combined_embeddings = np.concatenate((embeddings1, embeddings2,embeddings3))
combined_names = np.concatenate((names1, names2, names3))

# Save the combined embeddings and names to a new .npz file
np.savez('com.npz', combined_embeddings, combined_names)
