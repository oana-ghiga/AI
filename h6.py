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
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    # forward propagation
    def forward(self, x):
        x = self.hidden(x) #only one hidden layer
        x = self.relu(x) #reLU for input
        x = self.output(x)
        x = self.softmax(x) #softmax for output
        return x


# initializarea parametrilor
input_size = 7
hidden_size = 5
output_size = 3
learning_rate = 0.01
epochs = 100


# crearea unei instante a modelului + initializarea parametrilor
model = NeuralNet(input_size, hidden_size, output_size)

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

# Visualize misclassified points
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
hidden_weights = np.random.randn(input_size, hidden_size)
hidden_bias = np.zeros(hidden_size)
output_weights = np.random.randn(hidden_size, output_size)
output_bias = np.zeros(output_size)

# Forward pass for the hidden layer
hidden_output = np.dot(test_features, hidden_weights) + hidden_bias
hidden_activation = np.maximum(hidden_output, 0)  # ReLU activation

# Forward pass for the output layer
output = np.dot(hidden_activation, output_weights) + output_bias
exp_scores = np.exp(output)
output_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax for probabilities

# Manual backpropagation
probs = output_probs[np.arange(len(test_labels)), test_labels]
loss = -np.log(probs).mean()  # Cross-entropy loss

# Backpropagation for the output layer
grad_output = output_probs.copy()
grad_output[np.arange(len(test_labels)), test_labels] -= 1
grad_output /= len(test_labels)

output_weights_grad = np.dot(hidden_activation.T, grad_output)
output_bias_grad = np.sum(grad_output, axis=0)

# Backpropagation for the hidden layer
grad_hidden = np.dot(grad_output, output_weights.T)
grad_hidden[hidden_output <= 0] = 0  # Derivative of ReLU

hidden_weights_grad = np.dot(test_features.T, grad_hidden)
hidden_bias_grad = np.sum(grad_hidden, axis=0)

# Update weights manually
learning_rate = 0.01
hidden_weights -= learning_rate * hidden_weights_grad
hidden_bias -= learning_rate * hidden_bias_grad
output_weights -= learning_rate * output_weights_grad
output_bias -= learning_rate * output_bias_grad

# Print updated weights for demonstration
print("Updated Weights - Hidden Layer:")
print(hidden_weights)
print("Updated Bias - Hidden Layer:")
print(hidden_bias)
print("Updated Weights - Output Layer:")
print(output_weights)
print("Updated Bias - Output Layer:")
print(output_bias)