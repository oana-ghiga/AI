import torch
import torch.nn as nn
import torch.optim as optim
import neurolab as nl
import numpy as np

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

# antrenarea modelului
for epoch in range(epochs):
    # forward pass
    output = model(train_features)
    loss = loss_fn(output, train_labels)

    # backpropagation si optimizare
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# testarea modelului
test_output = model(test_features)
_, predicted = torch.max(test_output, 1)
accuracy = (predicted == test_labels).sum().item() / len(test_labels)
print('accuracy:', accuracy)

# 80% training 20% testing
with open('results.txt', 'w') as f:
    for i in range(len(test_labels)):
        f.write(f'exemplul {i+1}: clasa reala = {test_labels[i]}, clasa prezisa = {predicted[i]}\n')
