import numpy as np

# Load the .npy file into memory
data = np.load('task2_X_test.npy')

# View the contents, shape, and data type
print(data)
print("Shape:", data.shape)
print("Data Type:", data.dtype)
