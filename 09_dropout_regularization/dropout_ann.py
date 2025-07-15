import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * (1 - sig)

# XOR data
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Hyperparameters
epochs = 10000
lr = 0.1
input_size = 2
hidden_size = 4
output_size = 1
dropout_rate = 0.5

#weight initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)

    
    dropout_mask = (np.random.rand(*a1.shape) > dropout_rate).astype(float)
    a1 *= dropout_mask
    a1 /= (1.0 - dropout_rate) 

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    loss = np.mean((y - a2)**2)

    dL_da2 = 2 * (a2 - y)
    d_a2_z2 = sigmoid_derivative(a2)
    dL_dz2 = dL_da2 * d_a2_z2
    dL_dW2 = np.dot(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

    dL_da1 = np.dot(dL_dz2, W2.T)
    dL_da1 *= dropout_mask  
    d_a1_z1 = sigmoid_derivative(a1)
    dL_dz1 = dL_da1 * d_a1_z1
    dL_dW1 = np.dot(X.T, dL_dz1)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    #weight updae
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1