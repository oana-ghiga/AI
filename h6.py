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
hidden_weights = torch.randn(input_size, hidden_size, requires_grad=True)
hidden_bias = torch.zeros(hidden_size, requires_grad=True)
output_weights = torch.randn(hidden_size, output_size, requires_grad=True)
output_bias = torch.zeros(output_size, requires_grad=True)

# Forward pass for the hidden layer
hidden_output = test_features.mm(hidden_weights) + hidden_bias
hidden_activation = hidden_output.clamp(min=0)  # ReLU activation

# Forward pass for the output layer
output = hidden_activation.mm(output_weights) + output_bias
output_probs = output.softmax(dim=1)  # Softmax for probabilities

# Manual backpropagation
loss = -torch.log(output_probs[range(len(test_labels)), test_labels]).mean()  # Cross-entropy loss

# Backpropagation for the output layer
grad_output = output_probs.clone()
grad_output[range(len(test_labels)), test_labels] -= 1
grad_output /= len(test_labels)

output_weights.grad = hidden_activation.t().mm(grad_output)
output_bias.grad = grad_output.sum(dim=0)

# Backpropagation for the hidden layer
grad_hidden = grad_output.mm(output_weights.t())
grad_hidden[hidden_output <= 0] = 0  # Derivative of ReLU
hidden_weights.grad = test_features.t().mm(grad_hidden)
hidden_bias.grad = grad_hidden.sum(dim=0)

# Update weights manually
learning_rate = 0.01
with torch.no_grad():
    hidden_weights -= learning_rate * hidden_weights.grad
    hidden_bias -= learning_rate * hidden_bias.grad
    output_weights -= learning_rate * output_weights.grad
    output_bias -= learning_rate * output_bias.grad

# Print updated weights for demonstration
print("Updated Weights - Hidden Layer:")
print(hidden_weights)
print("Updated Bias - Hidden Layer:")
print(hidden_bias)
print("Updated Weights - Output Layer:")
print(output_weights)
print("Updated Bias - Output Layer:")
print(output_bias)