import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# citirea datelor
data = np.loadtxt('seeds_dataset.txt')
np.random.shuffle(data)
features = data[:, :7]
labels = data[:, 7:].flatten()

# normalizarea datelor
features = (features - features.mean(axis=0)) / features.std(axis=0)

# crearea seturilor de date de antrenare si de testare 80% 20%
train_features = torch.tensor(features[:int(len(features) * 0.8)], dtype=torch.float32)
test_features = torch.tensor(features[int(len(features) * 0.8):], dtype=torch.float32)
train_labels = torch.tensor(labels[:int(len(labels) * 0.8)] - 1, dtype=torch.long)
test_labels = torch.tensor(labels[int(len(labels) * 0.8):] - 1, dtype=torch.long)

# definirea modelului
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    # forward propagation
    def forward(self, x):
        x = self.hidden1(x) # first hidden layer
        x = self.relu(x) # reLU for first hidden layer
        x = self.hidden2(x) # second hidden layer
        x = self.relu(x) # reLU for second hidden layer
        x = self.output(x)
        x = self.softmax(x) # softmax for output
        return x


# initializarea parametrilor
input_size = 7
hidden_size1 = 5
hidden_size2 = 5
output_size = 3
learning_rate = 0.01
epochs = 100


# crearea unei instante a modelului + initializarea parametrilor
model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)

# definirea functiei de pierdere si a optimizatorului
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), learning_rate)

# Keep track of errors on training and testing sets
train_errors = []
test_errors = []

# antrenarea modelului
for epoch in range(epochs):
    # forward pass
    output = model(train_features)
    loss = loss_fn(output, train_labels)

    # backpropagation si optimizare
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute error on training set
    train_errors.append(loss.item())

    # Compute error on testing set
    test_output = model(test_features)
    test_loss = loss_fn(test_output, test_labels)
    test_errors.append(test_loss.item())

# Plot errors as a function of epochs
plt.figure(figsize=(10, 5))
plt.plot(train_errors, label='Training error')
plt.plot(test_errors, label='Testing error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()


# testarea modelului
test_output = model(test_features)
_, predicted = torch.max(test_output, 1)

# Compute accuracy
accuracy = (predicted == test_labels).sum().item() / len(test_labels)
print('accuracy:', accuracy)

# misclassified points
misclassified_indices = (predicted != test_labels).nonzero(as_tuple=True)[0]
misclassified_features = test_features[misclassified_indices]
misclassified_labels = test_labels[misclassified_indices]
misclassified_predictions = predicted[misclassified_indices]

for i in range(len(misclassified_indices)):
    print(f'Point {misclassified_indices[i]}: true label = {misclassified_labels[i]}, predicted label = {misclassified_predictions[i]}')
    print(f'Features: {misclassified_features[i]}')


# 80% training 20% testing
with open('results.txt', 'w') as f:
    for i in range(len(test_labels)):
        f.write(f'exemplul {i+1}: clasa reala = {test_labels[i]}, clasa prezisa = {predicted[i]}\n')

# Manual forward propagation
hidden_weights1 = np.random.randn(input_size, hidden_size1)
hidden_bias1 = np.zeros(hidden_size1)
hidden_weights2 = np.random.randn(hidden_size1, hidden_size2)
hidden_bias2 = np.zeros(hidden_size2)
output_weights = np.random.randn(hidden_size2, output_size)
output_bias = np.zeros(output_size)

# Forward pass for the first hidden layer
hidden_output1 = np.dot(test_features, hidden_weights1) + hidden_bias1
hidden_activation1 = np.maximum(hidden_output1, 0)  # ReLU

# Forward pass for the second hidden layer
hidden_output2 = np.dot(hidden_activation1, hidden_weights2) + hidden_bias2
hidden_activation2 = np.maximum(hidden_output2, 0)  # ReLU

# Forward pass for the output layer
output = np.dot(hidden_activation2, output_weights) + output_bias
exp_scores = np.exp(output)
output_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax

# Manual backpropagation
probs = output_probs[np.arange(len(test_labels)), test_labels]
loss = -np.log(probs).mean()  # Cross-entropy loss

# Backpropagation for the output layer
grad_output = output_probs.copy()
grad_output[np.arange(len(test_labels)), test_labels] -= 1
grad_output /= len(test_labels)

output_weights_grad = np.dot(hidden_activation2.T, grad_output)
output_bias_grad = np.sum(grad_output, axis=0)

# Backpropagation for the second hidden layer
grad_hidden2 = np.dot(grad_output, output_weights.T)
grad_hidden2[hidden_output2 <= 0] = 0

hidden_weights2_grad = np.dot(hidden_activation1.T, grad_hidden2)
hidden_bias2_grad = np.sum(grad_hidden2, axis=0)

# Backpropagation for the first hidden layer
grad_hidden1 = np.dot(grad_hidden2, hidden_weights2.T)
grad_hidden1[hidden_output1 <= 0] = 0

hidden_weights1_grad = np.dot(test_features.T, grad_hidden1)
hidden_bias1_grad = np.sum(grad_hidden1, axis=0)

# Update weights manually
learning_rate = 0.01
hidden_weights1 -= learning_rate * hidden_weights1_grad
hidden_bias1 -= learning_rate * hidden_bias1_grad
hidden_weights2 -= learning_rate * hidden_weights2_grad
hidden_bias2 -= learning_rate * hidden_bias2_grad
output_weights -= learning_rate * output_weights_grad
output_bias -= learning_rate * output_bias_grad

# Print updated weights for demonstration
print("Updated Weights - First Hidden Layer:")
print(hidden_weights1)
print("Updated Bias - First Hidden Layer:")
print(hidden_bias1)
print("Updated Weights - Second Hidden Layer:")
print(hidden_weights2)
print("Updated Bias - Second Hidden Layer:")
print(hidden_bias2)
print("Updated Weights - Output Layer:")
print(output_weights)
print("Updated Bias - Output Layer:")
print(output_bias)
